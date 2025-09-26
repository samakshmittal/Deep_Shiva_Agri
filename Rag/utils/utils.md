# Rag/utils Directory Detailed Function README

This README provides detailed explanations of each function defined in the code files within the `Rag/utils` directory of the [Deep_Shiva_Agri](https://github.com/samakshmittal/Deep_Shiva_Agri) repository. Each function description covers its purpose, inputs, outputs, and key implementation details.

---

## chunking.py

### `prepare_chunks_from_dataset(dataset_path)`
- **Purpose:** Loads a simple JSON dataset, validates each record (must have `id`, `text`, and `metadata`), and returns the chunks in a format suitable for Qdrant.
- **Inputs:** `dataset_path` (str) – Path to the dataset JSON file.
- **Outputs:** List of chunks, each a dict with `id`, `text`, and `metadata`.
- **Details:** Logs errors if the file is missing or records are incomplete.

### `chunking_runnable()`
- **Purpose:** Returns a LangChain `RunnableLambda` that wraps `_chunking_runnable_impl`.
- **Usage:** Designed for pipeline integration.

### `_chunking_runnable_impl(inputs)`
- **Purpose:** Implements chunking logic for use in a pipeline.
- **Inputs:** `inputs` (dict) – Must include `dataset_path`.
- **Outputs:** Dict containing all original inputs plus a `"chunks"` key with the prepared chunks.
- **Error Handling:** Returns empty chunks and logs on failure.

---

## create_embeding.py

### `chunk_data(raw_data, chunk_size=500, chunk_overlap=100)`
- **Purpose:** Splits raw dataset records into smaller text chunks using a recursive character splitter, preserving metadata and assigning unique numeric IDs.
- **Inputs:** `raw_data` (list of dicts), `chunk_size`, `chunk_overlap`.
- **Outputs:** Chunked data as list of dicts with `id`, `text`, and enhanced `metadata`.
- **Details:** Uses separators like newlines and punctuation for splitting.

### `embed_text_chain_fn(inputs)`
- **Purpose:** Reads raw JSON, chunks data, generates embeddings, saves them, and returns the processed data.
- **Inputs:** `inputs` (dict) – Must contain `raw_json_path` and optionally `embed_json_path`.
- **Outputs:** Dict with `chunks`, `embed_data`, and all original inputs.
- **Error Handling:** Raises on missing files or embedding errors.

### `embed_text_runnable()`
- **Purpose:** Returns a LangChain `RunnableLambda` for the embedding pipeline.
- **Usage:** Pipeline integration for embedding creation.

---

## llm.py

### `init_groq_client()`
- **Purpose:** Initializes and returns a Groq LLM client using an API key from environment variables.
- **Error Handling:** Raises ValueError if `GROQ_API_KEY` is not set.

### `get_groq_response(prompt, model="llama-3.1-8b-instant", temperature=0.2)`
- **Purpose:** Queries Groq’s LLM for a completion based on the prompt.
- **Inputs:** Prompt text, optional model name, and temperature.
- **Outputs:** Model response as a string.
- **Error Handling:** Returns error message string on failure.

### `build_prompt(query, chunks, max_words=4000)`
- **Purpose:** Composes an agriculture-specific prompt for the LLM, injecting retrieved document chunks and detailed instructions.
- **Inputs:** Query string, context chunks, max word limit.
- **Outputs:** A `PromptTemplate` object for LangChain.

---

## log.py

### `setup_logger(name="pipeline_logger", log_file="pipeline.log", level=logging.INFO)`
- **Purpose:** Configures and returns a logger that writes to file and console.
- **Inputs:** Logger name, log file path, logging level.
- **Outputs:** Configured logger instance.
- **Details:** Ensures handlers are only added once.

---

## qdrant.py

### `upload_in_batches(client, collection_name, points, batch_size=50)`
- **Purpose:** Uploads data points to Qdrant in batches, logging progress and errors.
- **Inputs:** Qdrant client, collection name, list of points, batch size.
- **Outputs:** None (side effect: uploads points).

### `upload_embed_to_qdrant(json_path, collection_name, qdrant_url, qdrant_api_key=None, vector_size=None)`
- **Purpose:** Loads embedded chunks from JSON and uploads them to a Qdrant collection, creating the collection if needed.
- **Inputs:** JSON path, collection name, Qdrant URL, optional API key and vector size.
- **Outputs:** Qdrant client instance.
- **Details:** Ensures unique point IDs and auto-detects vector size if unspecified.

### `search_qdrant(query_text, client, collection_name, top_k=5)`
- **Purpose:** Performs a similarity search in Qdrant for the most relevant chunks to a query.
- **Inputs:** Query text, Qdrant client, collection name, top-K.
- **Outputs:** List of top search results (chunks).
- **Details:** Uses local embedding model to encode the query.

### `rag_query(client, collection_name, query_text, top_k=5)`
- **Purpose:** Full RAG pipeline: searches Qdrant, builds prompt, gets LLM response, and returns both answer and context.
- **Inputs:** Qdrant client, collection name, query text, top-K.
- **Outputs:** Dict with `"response"` (LLM answer) and `"contexts"` (search results).

### `upload_qdrant_runnable()`
- **Purpose:** Returns a LangChain `RunnableLambda` for uploading to Qdrant.

### `_upload_qdrant_runnable_impl(inputs)`
- **Purpose:** Pipeline implementation for uploading embeddings to Qdrant.

### `rag_query_runnable()`
- **Purpose:** Returns a LangChain `RunnableLambda` for the RAG pipeline.

### `_rag_query_runnable_impl(inputs)`
- **Purpose:** Pipeline implementation for running a RAG query.

---

## Directory Usage

- **Integration:** These utilities are designed for use with LangChain pipelines and Qdrant vector database for agricultural RAG applications.
- **CLI Entrypoints:** Several files offer command line interfaces for manual execution.

---

> For further details, see the source code and inline docstrings in each file.
