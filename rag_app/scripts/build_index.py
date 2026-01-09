#!/usr/bin/env python3
"""Offline FAISS index builder.

Usage:
  - Set environment variable `OPENROUTER_API_KEY` before running.
  - Run this script from the project (rag_app) directory or directly with
    Python. It resolves paths relative to this script's parent directory.

What it does:
  - Loads chunks via `load_and_chunk_text` from `../data/inextlabs.txt`.
  - Generates embeddings with `embed_texts`.
  - Builds a FAISS IndexFlatL2 and adds embeddings.
  - Saves `embeddings/faiss_index.bin` and `embeddings/metadata.pkl`.
"""
from pathlib import Path
import sys
import pickle
import os

import numpy as np
import faiss

# Make sure we can import the local `app` package by adding rag_app/ to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.embeddings import load_and_chunk_text, embed_texts

#ONE TIME INDEX BUILDING SCRIPT (KNOWLEDGE BASE TO VECTOR STORE)
def main() -> None:
    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable not set.")
        print("Please set it with: export OPENROUTER_API_KEY='your-key-here'")
        sys.exit(1)


    #LOADS DATA FROM KNOWLEDGE BASE
    data_file = ROOT / "data" / "inextlabs.txt" 
    embeddings_dir = ROOT / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Check if data file exists
    if not data_file.exists():
        print(f"ERROR: Data file not found at {data_file}")
        sys.exit(1)

    # LOADING TEXT + CHUNKING USING embeddings.py FUNCTION (load_and_chunk_text, embed_texts)
    print("Loading and chunking text...") 
    chunks = load_and_chunk_text(str(data_file)) #embeddings.py function call
    if not chunks:
        print("No chunks produced from input file. Exiting.")
        return

    print(f"Loaded {len(chunks)} chunks")

    # EMBEDDING ALL CHUNKS
    print(f"Generating embeddings for {len(chunks)} chunks...")
    try:
        emb_arr = embed_texts(chunks) #embeddings.py function call
    except Exception as e:
        print(f"ERROR generating embeddings: {e}")
        print("\nPossible issues:")
        print("1. Check your OPENROUTER_API_KEY is valid")
        print("2. Verify you have credits in your OpenRouter account")
        print("3. Check your internet connection")
        sys.exit(1)

    # Validate embedding output
    if emb_arr is None:
        print("ERROR: embed_texts returned None")
        sys.exit(1)
    
    if not isinstance(emb_arr, np.ndarray):
        print(f"ERROR: embed_texts must return a NumPy array, got {type(emb_arr)}")
        sys.exit(1)
    
    if emb_arr.ndim != 2:
        print(f"ERROR: Expected 2D array, got shape {emb_arr.shape}")
        sys.exit(1)
    
    if emb_arr.dtype != np.float32:
        print(f"Converting embeddings from {emb_arr.dtype} to float32")
        emb_arr = emb_arr.astype(np.float32)

    n, d = emb_arr.shape
    print(f"Embedding shape: ({n}, {d})")

    if n != len(chunks):
        print(f"WARNING: Number of embeddings ({n}) doesn't match chunks ({len(chunks)})")

    # Build FAISS index (L2) SEMANTIC SEARCH (EUCLIDEAN DISTANCE)
    print("Building FAISS IndexFlatL2...")
    index = faiss.IndexFlatL2(d)
    index.add(emb_arr)

    index_file = embeddings_dir / "faiss_index.bin" #SAVING INDEX (CHUNK_ID:VECTOR MAPPING) VECTOR DATABASE
    metadata_file = embeddings_dir / "metadata.pkl" #SAVING CHUNKS (CHUNK_ID:TEXT MAPPING)  CHUNK TEXT DATABASE

    #SAVING INDEX (CHUNK_ID:VECTOR MAPPING)
    print(f"Saving FAISS index to {index_file}...")
    faiss.write_index(index, str(index_file))

    #SAVING CHUNKS (CHUNK_ID:TEXT MAPPING)
    print(f"Saving metadata (chunks) to {metadata_file}...")
    with open(metadata_file, "wb") as f:
        pickle.dump(chunks, f)

    print("\n✓ Index build complete!")
    print(f"  - Index: {index_file}")
    print(f"  - Metadata: {metadata_file}")
    print(f"  - Vectors: {n}")
    print(f"  - Dimensions: {d}")


if __name__ == "__main__":
    main()