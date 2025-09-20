import os
import json
from sentence_transformers import SentenceTransformer
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ------------------------------
# Load env vars
# ------------------------------
load_dotenv()

# ------------------------------
# Global Models
# ------------------------------
_local_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ------------------------------
# Chunking
# ------------------------------
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_data(raw_data, chunk_size=500, chunk_overlap=100):
    """
    Splits raw dataset into chunks using a recursive splitter
    while preserving metadata and assigning numeric IDs.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?"]
    )

    chunks = []
    chunk_counter = 1  # numeric ID counter

    for item in raw_data:
        text = item["text"]
        metadata = item.get("metadata", {})
        original_id = item["id"]

        # Use recursive splitter to get meaningful chunks
        split_texts = splitter.split_text(text)

        for idx, sub_text in enumerate(split_texts, start=1):
            chunks.append({
                "id": chunk_counter,
                "text": sub_text,
                "metadata": {
                    **metadata,
                    "tag": f"{original_id}_part{idx}" if len(split_texts) > 1 else original_id
                }
            })
            chunk_counter += 1

    return chunks


# ------------------------------
# Embedding (Dataset)
# ------------------------------
def embed_text_chain_fn(inputs):
    """
    Reads raw JSON → chunks (using recursive splitter) → embeddings → writes to embed_json_path.
    """
    raw_json_path = inputs.get("raw_json_path")
    embed_json_path = inputs.get("embed_json_path")

    if not raw_json_path or not os.path.exists(raw_json_path):
        raise ValueError("[embed_text_chain_fn] ❌ raw_json_path missing or invalid")

    # ✅ Load dataset
    with open(raw_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # ✅ Chunk dataset using recursive splitter
    chunks = chunk_data(raw_data)

    texts = [chunk["text"] for chunk in chunks]
    embed_data = []

    try:
        # Default: local MiniLM embeddings
        embeddings = _local_model.encode(
            texts, show_progress_bar=True, convert_to_numpy=True
        )
        for chunk, embedding in zip(chunks, embeddings):
            embed_data.append({
                "id": chunk["id"],                # numeric
                "text": chunk["text"],
                "embedding": embedding.tolist(),
                "metadata": chunk.get("metadata", {})
            })

        # ✅ Save embeddings JSON for Qdrant uploader
        if embed_json_path:
            os.makedirs(os.path.dirname(embed_json_path), exist_ok=True)
            with open(embed_json_path, "w", encoding="utf-8") as f:
                json.dump(embed_data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        raise RuntimeError(f"[❌] Error embedding text chunks: {e}")

    return {**inputs, "chunks": chunks, "embed_data": embed_data}


# ------------------------------
# Runnables
# ------------------------------
def embed_text_runnable():
    return RunnableLambda(embed_text_chain_fn)

