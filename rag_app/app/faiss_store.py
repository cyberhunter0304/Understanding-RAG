"""FAISS index loading and similarity search utilities."""

from pathlib import Path
import pickle
from typing import List, Tuple

import numpy as np
import faiss

from app.embeddings import embed_texts


class FaissStore:
    """
    Read-only FAISS vector store wrapper.

    Responsibilities:
    - Load FAISS index from disk
    - Load associated metadata (text chunks)
    - Embed query text
    - Perform top-K similarity search
    """
    #Loads FAISS index + METADATA
    def __init__(self, root_dir: Path):
        embeddings_dir = root_dir / "embeddings"

        index_path = embeddings_dir / "faiss_index.bin"
        metadata_path = embeddings_dir / "metadata.pkl"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))

        # Load metadata (chunks)
        with open(metadata_path, "rb") as f:
            self.chunks: List[str] = pickle.load(f)

        # Sanity check
        if self.index.ntotal != len(self.chunks):
            raise RuntimeError(
                "FAISS index size does not match metadata size "
                f"({self.index.ntotal} vs {len(self.chunks)})"
            )

    #Similarity Search - Embedding User Query + FAISS Search
    def similarity_search(
        self, query: str, k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Perform similarity search against the FAISS index.

        Args:
            query: User query string
            k: Number of results to return

        Returns:
            List of (chunk_text, distance) tuples
        """
        if not query.strip():
            return []

        # Embed query
        query_emb = embed_texts([query]) #embeddings.py function call
        if not isinstance(query_emb, np.ndarray):
            raise RuntimeError("embed_texts must return a NumPy array")
        if query_emb.dtype != np.float32:
            query_emb = query_emb.astype(np.float32)

        # FAISS search
        distances, indices = self.index.search(query_emb, k)

        results: List[Tuple[str, float]] = [] #List of (chunk_text, distance) tuples
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0:
                continue
            results.append((self.chunks[idx], float(dist)))

        return results #Top 5 Results 
        
