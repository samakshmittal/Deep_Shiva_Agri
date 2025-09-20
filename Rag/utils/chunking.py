import os
import json
from utils.log import setup_logger
from langchain_core.runnables import RunnableLambda

logger = setup_logger()

def prepare_chunks_from_dataset(dataset_path):
    """
    Loads a simple JSON dataset in Qdrant-ready format.
    Each record must have: id, text, metadata.
    Returns the chunks directly.
    """
    if not os.path.exists(dataset_path):
        logger.error(f"❌ Dataset file not found: {dataset_path}")
        return []

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate format
    chunks = []
    for i, record in enumerate(data):
        if "id" not in record or "text" not in record or "metadata" not in record:
            logger.warning(f"⚠️ Skipping record {i}, missing required keys: {record}")
            continue

        chunks.append({
            "id": record["id"],
            "text": record["text"],
            "metadata": record["metadata"]
        })

    logger.info(f"✅ Loaded {len(chunks)} chunks from dataset")
    return chunks


# -------------------------------
# Runnable Wrapper
# -------------------------------
def chunking_runnable():
    return RunnableLambda(lambda inputs: _chunking_runnable_impl(inputs))


def _chunking_runnable_impl(inputs):
    try:
        dataset_path = inputs.get("dataset_path")
        if not dataset_path:
            raise ValueError("❌ Missing 'dataset_path' in inputs")

        chunks = prepare_chunks_from_dataset(dataset_path)

        return {
            **inputs,  # pass forward everything
            "chunks": chunks
        }

    except Exception as e:
        logger.exception("❌ Failed in chunking runnable")
        return {**inputs, "chunks": []}


# -------------------------------
# CLI Entrypoint
# -------------------------------
if __name__ == "__main__":
    print("🔧 Chunker CLI - Loading simple dataset!")

    dataset_path = input("📄 Enter path to dataset JSON: ").strip()
    if not os.path.exists(dataset_path):
        logger.error(f"❌ File not found: {dataset_path}")
        exit(1)

    inputs = {"dataset_path": dataset_path}
    runnable = chunking_runnable()
    output = runnable.invoke(inputs)

    out_file = os.path.splitext(os.path.basename(dataset_path))[0] + "_chunks.json"
    out_path = os.path.join("RAG_TENDOR/temp_uploads", out_file)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as out:
        json.dump(output["chunks"], out, indent=2, ensure_ascii=False)

    logger.info(f"💾 Chunks saved to: {out_path}")
    print(f"✅ All done! Output written to: {out_path}")
