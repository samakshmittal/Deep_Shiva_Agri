import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# -------------------------
# RMSNorm
# -------------------------
class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm (no centering)"""
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        # x: [batch, seq, dim]
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        x_normed = x / rms
        return x_normed * self.scale


# -------------------------
# Rotary positional embeddings (RoPE)
# -------------------------
def rotate_every_two(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q/k shape: [batch, heads, seq, head_dim]
    # cos/sin shape: [seq, head_dim]
    # apply to first head_dim dims
    q_ = (q * cos) + (rotate_every_two(q) * sin)
    k_ = (k * cos) + (rotate_every_two(k) * sin)
    return q_, k_


def fixed_sincos_pos_emb(seq_len: int, dim: int, device=None):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i , j -> i j", t, inv_freq)  # [seq, dim/2]
    emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)  # not used directly, we'll produce cos/sin
    # Instead produce cos and sin interleaved to match rotate_every_two behavior
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    return cos, sin  # shapes: [seq, dim]


# -------------------------
# Multi-Query Attention (MQA) variant
# -------------------------
class MultiQueryAttention(nn.Module):
    def __init__(self, dim, n_query_heads=8, n_kv_heads=2, head_dim=None, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.nq = n_query_heads
        self.nkv = n_kv_heads
        self.head_dim = head_dim or (dim // n_query_heads)
        assert self.dim % self.nq == 0, "dim must be divisible by number of query heads"
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # projections
        self.q_proj = nn.Linear(dim, self.nq * self.head_dim, bias=False)
        # fewer kv heads
        self.k_proj = nn.Linear(dim, self.nkv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.nkv * self.head_dim, bias=False)

        self.out_proj = nn.Linear(self.nq * self.head_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None, cos=None, sin=None):
        # x: [batch, seq, dim]
        B, S, _ = x.shape

        q = self.q_proj(x)  # [B, S, nq * head_dim]
        k = self.k_proj(x)  # [B, S, nkv * head_dim]
        v = self.v_proj(x)  # [B, S, nkv * head_dim]

        # reshape
        q = q.view(B, S, self.nq, self.head_dim).transpose(1, 2)  # [B, nq, S, head_dim]
        k = k.view(B, S, self.nkv, self.head_dim).transpose(1, 2)  # [B, nkv, S, head_dim]
        v = v.view(B, S, self.nkv, self.head_dim).transpose(1, 2)  # [B, nkv, S, head_dim]

        # replicate k,v across query heads (multi-query): expand nkv -> nq
        if self.nkv != self.nq:
            k = k.repeat(1, self.nq // self.nkv + (1 if self.nq % self.nkv else 0), 1, 1)
            v = v.repeat(1, self.nq // self.nkv + (1 if self.nq % self.nkv else 0), 1, 1)
            k = k[:, :self.nq, ...]
            v = v[:, :self.nq, ...]
        # now k/v shapes: [B, nq, S, head_dim]

        # apply rotary (RoPE) if provided - cos/sin: [S, head_dim]
        if cos is not None and sin is not None:
            # expand cos/sin to [1, 1, S, head_dim]
            cos_b = cos.unsqueeze(0).unsqueeze(0)
            sin_b = sin.unsqueeze(0).unsqueeze(0)
            q, k = apply_rotary_pos_emb(q, k, cos_b, sin_b)

        # attention
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, nq, S, S]
        if attn_mask is not None:
            attn_logits = attn_logits + attn_mask  # attn_mask should be additive logit mask

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # [B, nq, S, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, S, self.nq * self.head_dim)  # [B, S, dim_q]
        out = self.out_proj(out)  # [B, S, dim]
        return out


# -------------------------
# SwiGLU Feedforward
# -------------------------
class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        # project to 2*hidden, then split
        self.w1 = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        # x: [B, S, dim]
        x_proj = self.w1(x)  # [B, S, 2*hidden]
        a, b = x_proj.chunk(2, dim=-1)
        return self.w2(F.silu(a) * b)


# -------------------------
# Transformer Block
# -------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, n_query_heads=8, n_kv_heads=2, ffn_hidden=1536, dropout=0.0):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = MultiQueryAttention(dim, n_query_heads=n_query_heads, n_kv_heads=n_kv_heads, dropout=dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_hidden)

    def forward(self, x, cos=None, sin=None, attn_mask=None):
        # attention block with residual
        x_attn = self.attn_norm(x)
        attn_out = self.attn(x_attn, attn_mask=attn_mask, cos=cos, sin=sin)
        x = x + self.resid_dropout(attn_out)

        # ffn block
        x_ffn = self.ffn_norm(x)
        ffn_out = self.ffn(x_ffn)
        x = x + self.resid_dropout(ffn_out)

        return x


# -------------------------
# Full Model
# -------------------------
class SmallLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,           # token embedding dim (image uses 512)
        n_layers: int = 12,       # repeat n times
        n_query_heads: int = 8,   # 8 query heads
        n_kv_heads: int = 2,      # 2 key-value heads
        ffn_hidden: int = 1536,   # hidden dim of feed-forward (SwiGLU)
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        tie_word_embeddings: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_dropout = nn.Dropout(dropout)

        # rotary parameters (we create cos/sin per model forward if needed)
        # use half head dim for rotary? We'll use full head_dim
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_query_heads=n_query_heads, n_kv_heads=n_kv_heads, ffn_hidden=ffn_hidden, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_rms = RMSNorm(dim)

        # output projection (tied to token_emb weights if requested)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        if tie_word_embeddings:
            # tie weights
            self.lm_head.weight = self.token_emb.weight

        # save rotary head dim for cos/sin generation (head_dim)
        self.head_dim = dim // n_query_heads
        self.n_query_heads = n_query_heads

    def get_rotary_cos_sin(self, seq_len: int, device):
        # produce cos and sin shaped [seq_len, head_dim]
        cos, sin = fixed_sincos_pos_emb(seq_len, self.head_dim, device=device)
        return cos, sin

    def forward(self, input_ids: torch.LongTensor, attn_mask: Optional[torch.Tensor] = None):
        """
        input_ids: [batch, seq]
        attn_mask: optional additive mask to add to attention logits, shape broadcastable to [batch, heads, seq, seq]
                   (e.g. causal mask with -inf for disallowed positions)
        """
        B, S = input_ids.size()
        assert S <= self.max_seq_len, "Sequence length exceeds model max_seq_len"

        # embeddings
        x = self.token_emb(input_ids)  # [B, S, dim]

        # rotary cos/sin
        cos, sin = self.get_rotary_cos_sin(S, device=x.device)  # [S, head_dim]

        x = self.pos_dropout(x)

        # pass through blocks
        for blk in self.blocks:
            x = blk(x, cos=cos, sin=sin, attn_mask=attn_mask)

        x = self.final_rms(x)
        logits = self.lm_head(x)  # [B, S, vocab]
        return logits


# -------------------------
# Causal mask helper
# -------------------------
def make_causal_mask(batch_size: int, seq_len: int, device=None):
    # additive mask with -inf for future positions, 0 for allowed
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float("-inf"), diagonal=1)
    # expand to [batch, heads, seq, seq] if needed by attention; but our attention expects additive mask broadcastable
    return mask.unsqueeze(0).unsqueeze(0)  # [1,1,seq,seq]


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # quick instantiate and test with random tokens
    BATCH = 2
    SEQ = 128
    VOCAB = 30000

    model = SmallLM(vocab_size=VOCAB, dim=512, n_layers=12, n_query_heads=8, n_kv_heads=2, ffn_hidden=1536, max_seq_len=2048)
    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ))
    causal_mask = make_causal_mask(BATCH, SEQ, device=input_ids.device)
    logits = model(input_ids, attn_mask=causal_mask)  # [B, S, VOCAB]
    print("Logits shape:", logits.shape)
