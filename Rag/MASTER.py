import os
import sys
from dotenv import load_dotenv
from collections import deque

MAX_TURNS = 10
chat_history = deque(maxlen=MAX_TURNS)

# Only embedding + Qdrant needed now
from utils.create_embeding import embed_text_runnable
from utils.qdrant import upload_qdrant_runnable, rag_query_runnable
from utils.log import setup_logger
from utils.llm import build_prompt, get_groq_response


# --------------------------------
# ⚙️ Setup
# --------------------------------
logger = setup_logger(__name__)
load_dotenv()

# ------------------------------
# 🔁 New RAG Pipeline: JSON → Embed → Qdrant
# ------------------------------
rag_pipeline = (
    embed_text_runnable()
    | upload_qdrant_runnable()
)

query_pipe = rag_query_runnable()


# ------------------------------
# 📥 Pipeline Execution Function
# ------------------------------
def run_rag_pipeline(input_dataset_path, collection_name="my_dataset"):
    """
    Runs the ingestion pipeline directly on a JSON dataset file.
    """
    file_name = os.path.splitext(os.path.basename(input_dataset_path))[0]
    input_dict = {
        "raw_json_path": input_dataset_path,   # ✅ Direct JSON (already structured)
        "embed_json_path": f"./temp_uploads/{file_name}_embeddings.json",
        "source_name": file_name,
        "title": "Custom Dataset",
        "collection_name": collection_name
    }

    logger.info("🔧 Running dataset ingestion → embedding → Qdrant upload...")
    result = rag_pipeline.invoke(input_dict)
    return result


def add_user_message(message_list, text):
    user_message = {
        "role": "user",
        "content": text
    }
    message_list.append(user_message)


def add_assistant_message(message_list, text):
    asst_message = {
        "role": "assistant",
        "content": text
    }
    message_list.append(asst_message)


# # ------------------------------
# # 💬 Chat Loop
# # ------------------------------
# def chat_loop(qdrant_client, collection_name):
#     print("\n💬 You can now ask questions! Type 'exit' to quit.\n")
#     while True:
#         query = input("🧠 Ask your query: ").strip()
#         if query.lower() in {"exit", "quit"}:
#             print("👋 Goodbye!")
#             break

#         try:
#             query_inputs = {
#                 "qdrant_client": qdrant_client,
#                 "collection_name": collection_name,
#                 "query": query,
#                 "history": list(chat_history)
#             }
#             response = query_pipe.invoke(query_inputs)
#             answer = response["response"]
#             contexts = response["contexts"]

#             print("🤖", answer, "\n", contexts)
#             chat_history.append([query, answer, contexts])

#         except Exception as e:
#             logger.exception("❌ Query failed.")
#             print(f"❌ Error: {str(e)}\n")


# ------------------------------
# 💬 Chat Loop
# ------------------------------
def chat_loop(qdrant_client, collection_name):
    print("\n💬 You can now ask questions! Type 'exit' to quit.\n")
    total_usage = 0
    messages = []

    while True:
        query = input("🧠 Ask your query: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        try:
            # 1️⃣ Save user input
            add_user_message(messages, query)

            # 2️⃣ Retrieve relevant chunks from Qdrant
            query_inputs = {
                "qdrant_client": qdrant_client,
                "collection_name": collection_name,
                "query": query,
                "history": list(chat_history)  # optional short history
            }
            response = query_pipe.invoke(query_inputs)
            retrieved_chunks = response["contexts"]

            # 3️⃣ Build agriculture-specific prompt
            prompt_template = build_prompt(query, retrieved_chunks)
            prompt_text = prompt_template.format()  # convert PromptTemplate → string

            # 4️⃣ Get LLM response from Groq
            answer = get_groq_response(prompt_text)

            # 5️⃣ Save assistant response
            add_assistant_message(messages, answer)

            # 6️⃣ Update token usage (optional / dummy)
            if "usage" in response:
                usage = response["usage"]["input_tokens"] + response["usage"]["output_tokens"]
                total_usage += usage
            else:
                usage = 0

            # 7️⃣ Print outputs
            print("\n🤖 Answer:\n", answer)
            print("\n📚 Contexts used:")
            for chunk in retrieved_chunks:
                print(f"- ID={chunk.get('id')}, Score={chunk.get('score', 'N/A')}, Meta={chunk.get('metadata')}")
            print("\n📊 Total usage till now:", total_usage)

            # 8️⃣ Save to chat history
            chat_history.append([query, answer, retrieved_chunks])

        except Exception as e:
            logger.exception("❌ Query failed.")
            print(f"❌ Error: {str(e)}\n")

# ------------------------------
# 🚀 Entry Point
# ------------------------------
if __name__ == "__main__":
    print("📚 Dataset-powered RAG System")
    dataset_path = input("📄 Enter dataset JSON path: ").strip()

    if not os.path.exists(dataset_path):
        print("❌ Dataset file not found.")
        sys.exit(1)

    try:
        ingest_result = run_rag_pipeline(dataset_path, collection_name="crop_dataset")
        if ingest_result.get("qdrant_client") is None:
            raise Exception("❌ Qdrant client not initialized.")

        chat_loop(
            qdrant_client=ingest_result["qdrant_client"],
            collection_name=ingest_result["collection_name"]
        )

    except Exception as e:
        logger.exception("❌ Fatal pipeline error.")
        print(f"❌ Error: {str(e)}")
