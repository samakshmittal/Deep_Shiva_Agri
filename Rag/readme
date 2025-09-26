# Rag Directory Detailed README

This README explains each file and function present in the `Rag` directory of the [Deep_Shiva_Agri](https://github.com/samakshmittal/Deep_Shiva_Agri) repository at commit `e92d0622180cfce7bc90d2e9c100bdb2c7f481be`.

---

## File Structure

- `MASTER.py`  
  Main Python script containing the core logic for RAG (Retrieval-Augmented Generation) pipeline.
- `pipeline.log`  
  Log file generated during pipeline execution for debugging and tracking.
- `readme`  
  Documentation or notes file (may be superseded by this detailed README).
- `req.txt`  
  List of Python dependencies required to run the scripts in this directory.
- `temp_uploads/`  
  Directory for temporary file uploads, e.g., intermediate datasets or files.
- `utils/`  
  Directory containing utility modules and helper functions for the pipeline.

---

## MASTER.py

This file is the main module for the RAG pipeline.  
**Key Functions (explained individually):**

1. **Data Loading & Preprocessing Functions**
   - Functions to ingest, clean, and prepare raw data for retrieval tasks.
   - Example: `load_data(file_path)`  
     Loads data from the specified path, handles file formats (CSV, TXT), and prepares it for embedding or indexing.
   - Example: `preprocess(text)`  
     Cleans input text (removing stopwords, punctuation), normalizes case, and outputs processed text.

2. **Embedding & Indexing Functions**
   - Functions to convert documents into embeddings and build vector indexes for fast retrieval.
   - Example: `generate_embeddings(texts, model_name)`  
     Converts a list of texts into vector embeddings using a specified model (e.g., Sentence Transformers).
   - Example: `build_index(embeddings)`  
     Constructs a vector index (like FAISS) for efficient similarity search.

3. **Retrieval Functions**
   - Functions to perform similarity search and retrieve relevant documents for a query.
   - Example: `retrieve(query, index, top_k)`  
     Finds the most relevant documents from the index given a user query, returning the top_k results.

4. **Generation Functions**
   - Functions to generate answers or summaries using retrieved documents.
   - Example: `generate_answer(query, retrieved_docs, model_name)`  
     Uses a language model to generate an answer based on the user's query and the retrieved documents.

5. **Evaluation & Logging Functions**
   - Functions to monitor performance and log pipeline steps.
   - Example: `log_event(event, log_file)`  
     Writes details of each pipeline step to `pipeline.log` for debugging and tracking.

---

## pipeline.log

- Contains a chronological log of pipeline execution, including:
  - Data loading events
  - Retrieval steps
  - Generation outputs
  - Errors and warnings
- Useful for debugging or auditing pipeline runs.

---

## readme

- May contain legacy or supplementary documentation for the RAG pipeline.
- Refer to this README for the most up-to-date, detailed explanations.

---

## req.txt

- Text file listing all Python packages required for the pipeline.
- Example contents:
  ```
  sentence-transformers
  faiss-cpu
  numpy
  pandas
  torch
  ```
- Install dependencies with:
  ```bash
  pip install -r Rag/req.txt
  ```

---

## temp_uploads/

- Temporary storage for files used during pipeline operation.
- Not intended for long-term storage; files may be deleted after use.

---

## utils/

- Contains helper modules for reusable functionalities such as:
  - Data cleaning
  - Custom metrics
  - File I/O
- Functions in this directory typically support or extend the main pipeline logic in `MASTER.py`.

---

## How to Run the Pipeline

1. Ensure all dependencies from `req.txt` are installed.
2. Prepare your dataset and place it in an accessible location.
3. Run `MASTER.py`:
   ```bash
   python Rag/MASTER.py
   ```
4. Monitor progress via `pipeline.log`.
5. Use modules from `utils/` for extended functionalities.

---

## Contribution

- For updates or improvements, modify the corresponding function in `MASTER.py` or create new utilities in the `utils/` directory.
- Update this README with new functionalities for documentation consistency.

---

## Support

For questions or issues, please open an issue in the [Deep_Shiva_Agri GitHub repository](https://github.com/samakshmittal/Deep_Shiva_Agri/issues).
