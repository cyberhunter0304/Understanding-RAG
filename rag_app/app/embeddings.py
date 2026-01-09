from typing import List
import re
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


#TEXT TO CHUNKS
def load_and_chunk_text(file_path: str) -> List[str]:
    """
    Chunk website-style text using word-based chunking.
    Designed for landing pages, marketing content, and UI-heavy text.
    """

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))  # words
    OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))   # words overlap for continuity

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words).strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - OVERLAP

    return chunks

#CHUNKS TO VECTORS + NORMALISED
def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of texts using OpenRouter's embedding API.
    
    - Uses google/gemini-embedding-001
    - Supports batch processing
    - Returns normalized embeddings compatible with FAISS
    """

    if not texts:
        return np.array([])
    
    try:
        embedding_response = client.embeddings.create(
            extra_headers={
                "HTTP-Referer": os.getenv("SITE_URL", ""),
                "X-Title": os.getenv("SITE_NAME", ""),
            },
            model=os.getenv("EMBEDDING_MODEL", "google/gemini-embedding-001"),
            input=texts,  # Batch processing supported
            encoding_format="float"
        )
        
        # Extract embeddings from response
        embeddings = [data.embedding for data in embedding_response.data]
        embeddings_array = np.asarray(embeddings, dtype=np.float32)
        
        # Normalize embeddings for cosine similarity (FAISS compatibility)
        # Value between 0 to 1 for better conclusions
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        normalized_embeddings = embeddings_array / norms
        
        return normalized_embeddings
    
    except Exception as e:
        print(f"Error creating embeddings: {str(e)}")
        return np.array([])

#Single Query Embedder (User)
def embed_single_text(text: str) -> np.ndarray:
    """
    Embed a single text string.
    Convenience function for single queries.
    """
    return embed_texts([text])[0] if text else np.array([])