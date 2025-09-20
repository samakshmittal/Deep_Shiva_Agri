import json
import sys
import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

from utils.log import setup_logger
from utils.llm import get_groq_response, build_prompt

# Setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()
logger = setup_logger("qdrant_logger")

# Default local model (used for queries if OpenAI not set)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ------------------------------
# Upload utilities
# ------------------------------
def upload_in_batches(client, collection_name, points, batch_size=50):
    """Uploads points to Qdrant in batches and logs each batch size."""
    total_points = len(points)
    total_batches = (total_points + batch_size - 1) // batch_size

    for i in range(0, total_points, batch_size):
        batch = points[i:i + batch_size]
        try:
            client.upsert(collection_name=collection_name, points=batch)
            logger.info(f"✅ Uploaded batch {i // batch_size + 1}/{total_batches} ({len(batch)} points)")
        except Exception as e:
            logger.error(f"❌ Error uploading batch {i // batch_size + 1}: {e}")

    logger.info(f"📊 Finished uploading {total_points} points to '{collection_name}' in {total_batches} batches.")


def upload_embed_to_qdrant(
    json_path, collection_name, qdrant_url, qdrant_api_key=None, vector_size=None
):
    """
    Uploads embedded chunks from JSON to Qdrant.
    Appends to existing collection if it exists.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"📄 Loaded {len(data)} chunks from {json_path}")

    # Infer vector size if not provided
    if not vector_size and data and "embedding" in data[0]:
        vector_size = len(data[0]["embedding"])
        logger.info(f"🔍 Auto-detected vector size: {vector_size}")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # Create collection if it doesn't exist
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info(f"🆕 Created collection '{collection_name}' with vector size {vector_size}")
        existing_count = 0
    else:
        existing_count = client.count(collection_name=collection_name, exact=True).count
        logger.info(f"⚡ Collection '{collection_name}' exists with {existing_count} points. Appending new points.")

    # Prepare points with unique IDs (start after existing_count)
    points = [
        PointStruct(
            id=existing_count + i,  # ensure unique ID
            vector=chunk["embedding"],
            payload={
                "dataset_id": chunk.get("id", str(existing_count + i)),
                "text": chunk["text"],
                "metadata": chunk.get("metadata", {})
            },
        )
        for i, chunk in enumerate(data)
    ]

    # Upload points in batches
    upload_in_batches(client, collection_name, points, batch_size=50)
    return client

# def upload_embed_to_qdrant(
#     json_path, collection_name, qdrant_url, qdrant_api_key=None, vector_size=None
# ):
#     """
#     Uploads embedded chunks from JSON to Qdrant.
#     Uses dataset's `id`, `text`, and `metadata` directly.
#     """
#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     logger.info(f"📄 Loaded {len(data)} chunks from {json_path}")

#     # Infer vector size if not provided
#     if not vector_size and data and "embedding" in data[0]:
#         vector_size = len(data[0]["embedding"])
#         logger.info(f"🔍 Auto-detected vector size: {vector_size}")

#     client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

#     # Ensure collection exists
#     if client.collection_exists(collection_name):
#         collection_info = client.get_collection(collection_name)
#         existing_config = collection_info.config.params.vectors

#         if (
#             hasattr(existing_config, "size")
#             and existing_config.size == vector_size
#             and existing_config.distance == Distance.COSINE
#         ):
#             existing_count = client.count(collection_name=collection_name, exact=True).count
#             if existing_count > 0:
#                 logger.info(
#                     f"✅ Collection '{collection_name}' already exists with {existing_count} points. Skipping upload."
#                 )
#                 return client
#             else:
#                 logger.info(f"📝 Collection '{collection_name}' exists but empty. Proceeding.")
#         else:
#             logger.warning(
#                 f"⚠️ Collection '{collection_name}' exists but has mismatched vector config. Skipping upload."
#             )
#             logger.warning(
#                 f"   Existing: size={existing_config.size}, distance={existing_config.distance}"
#             )
#             logger.warning(f"   Expected: size={vector_size}, distance={Distance.COSINE}")
#             return client
#     else:
#         client.create_collection(
#             collection_name=collection_name,
#             vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
#         )
#         logger.info(f"🆕 Created collection '{collection_name}' with vector size {vector_size}")

#     # Upload all chunks
#     points = [
#         PointStruct(
#             id=i,  # ✅ always use numeric index for Qdrant
#             vector=chunk["embedding"],
#             payload={
#                 "dataset_id": chunk.get("id", str(i)),  # ✅ keep your original ID here
#                 "text": chunk["text"],
#                 "metadata": chunk.get("metadata", {})
#             },
#         )
#         for i, chunk in enumerate(data)
#     ]

#     upload_in_batches(client, collection_name, points, batch_size=50)
#     return client


# ------------------------------
# Search & RAG
# ------------------------------
def search_qdrant(query_text, client: QdrantClient, collection_name: str, top_k: int = 5):
    """Search Qdrant for most similar chunks."""
    query_vector = model.encode([query_text], convert_to_numpy=True)

    result = client.search(
        collection_name=collection_name, query_vector=query_vector[0].tolist(), limit=top_k
    )
    logger.info(f"🔎 Retrieved {len(result)} results for query: '{query_text}'")

    for i, res in enumerate(result, 1):
        meta = res.payload.get("metadata", {})
        logger.info(
            f"Match {i}: ID={res.id}, Score={res.score:.4f}, Meta={meta}"
        )
    return result


def rag_query(client, collection_name, query_text, top_k=5):
    logger.info(f"Running RAG query for: {query_text}")
    try:
        chunks = search_qdrant(query_text, client, collection_name, top_k=top_k)
        if not chunks:
            return {"response": "No relevant info found.", "contexts": []}

        prompt = build_prompt(query_text, chunks)
        response = get_groq_response(prompt)

        context_texts = [
            {
                "id": chunk.id,
                "text": chunk.payload.get("text", ""),
                "metadata": chunk.payload.get("metadata", {}),
                "score": chunk.score
            }
            for chunk in chunks
        ]

        return {"response": response, "contexts": context_texts}
    except Exception as e:
        logger.exception(f"❌ Error in RAG query: {e}")
        return {"response": f"Error: {str(e)}", "contexts": []}



# ------------------------------
# Runnables
# ------------------------------
def upload_qdrant_runnable():
    return RunnableLambda(_upload_qdrant_runnable_impl)


def _upload_qdrant_runnable_impl(inputs):
    try:
        client = upload_embed_to_qdrant(
            json_path=inputs["embed_json_path"],
            collection_name=inputs["collection_name"],
            qdrant_url=inputs.get("qdrant_url") or os.getenv("QDRANT_URL"),
            qdrant_api_key=inputs.get("qdrant_api_key") or os.getenv("QDRANT_API_KEY"),
            vector_size=inputs.get("vector_size"),  # auto-detect if None
        )
        return {**inputs, "qdrant_client": client}
    except Exception as e:
        logger.exception(f"❌ Upload failed: {e}")
        return {**inputs, "qdrant_client": None, "error": str(e)}


def rag_query_runnable():
    return RunnableLambda(_rag_query_runnable_impl)


def _rag_query_runnable_impl(inputs):
    try:
        result = rag_query(
            client=inputs["qdrant_client"],
            collection_name=inputs["collection_name"],
            query_text=inputs["query"],
            top_k=inputs.get("top_k", 5),
        )
        return {**inputs, "response": result["response"], "contexts": result["contexts"]}
    except Exception as e:
        logger.exception(f"❌ Error in rag_query_runnable: {e}")
        return {**inputs, "response": f"[❌] RAG failed: {e}", "contexts": []}


# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 4:
        logger.error("Usage: python qdrant.py <json_path> <collection_name> <qdrant_url> [api_key]")
        sys.exit(1)

    json_path, collection_name, qdrant_url = sys.argv[1:4]
    qdrant_api_key = sys.argv[4] if len(sys.argv) > 4 else None

    try:
        client = upload_embed_to_qdrant(json_path, collection_name, qdrant_url, qdrant_api_key)
        test_query = "What are the main requirements?"
        answer = rag_query(client, collection_name, test_query)
        logger.info(f"Final Answer:\n{answer}")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
