# llm.py
import os
from groq import Groq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.prompts import PromptTemplate

load_dotenv()

# ------------------------------
# Initialize Groq client
# ------------------------------
def init_groq_client():
    """
    Initializes and returns a Groq client using the API key from environment variables.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in the environment variables")
    return Groq(api_key=api_key)


# ------------------------------
# Get response from Groq LLM
# ------------------------------
def get_groq_response(prompt, model="llama-3.1-8b-instant", temperature=0.2):
    """
    Get response from Groq's LLM using the official client.
    """
    try:
        client = init_groq_client()
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes and answers based only on the provided context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=model,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[❌] Error getting LLM response: {str(e)}"


# ------------------------------
# Build agriculture-specific prompt
# ------------------------------
def build_prompt(query, chunks, max_words=4000):
    """
    Builds a prompt for the agricultural assistant use case.
    Injects retrieved chunks and provides detailed instructions.
    """
    # Prepare context block
    context_texts = []
    total_words = 0

    for idx, chunk in enumerate(chunks):
        if hasattr(chunk, "payload"):
            text = chunk.payload.get("text", "")
            metadata = chunk.payload.get("metadata", {})
        elif isinstance(chunk, dict):
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
        else:
            text, metadata = str(chunk), {}

        # Use 'tag' from embedding metadata as chunk ID
        chunk_id = metadata.get("tag", idx)
        page_num = metadata.get("page_number", "N/A")

        words = text.split()
        if total_words + len(words) > max_words:
            break

        context_texts.append(f"[Chunk {chunk_id}, Page {page_num}]\n{text.strip()}")
        total_words += len(words)

    context_block = "\n\n".join(context_texts)

    # Prompt template
    template = f"""
You are an expert agricultural assistant with deep knowledge of farming, crops, livestock, weather, and agricultural technology. 
You help farmers and agricultural enthusiasts with their questions and concerns.

Your expertise includes:
- Crop management and cultivation techniques
- Pest and disease identification and treatment
- Soil health and fertilization
- Weather impacts on agriculture
- Livestock care and management
- Agricultural technology and equipment
- Sustainable farming practices
- Market information and pricing
- Regional agricultural practices in India

-------------------- CONTEXT START --------------------
Below are excerpts from the documents for reference.

{context_block}
--------------------- CONTEXT END ---------------------

Instructions:
- Read the context thoroughly.
- Answer the question in a detailed paragraph format.
- Cite chunks explicitly using the [Chunk X] labels from the context,
- Use only the information from the context.
- If the answer is not present, reply: "I don't know based on the provided information."
- Optionally, include a relevance or confidence score at the end.
- Optionally, include a reasoning path for the same.

Question:
{query}

Answer:
"""

    return PromptTemplate(template=template, input_variables=[])
