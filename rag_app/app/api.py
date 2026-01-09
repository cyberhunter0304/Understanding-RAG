from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.faiss_store import FaissStore
from app.llm import call_llm


ROOT = Path(__file__).resolve().parents[1]

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


@app.post("/chat")
def chat(req: QueryRequest):
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query must be a non-empty string")

    try:
        store = FaissStore(ROOT)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to load vector store: {e}")

    # Retrieve top-k chunks (k=5)
    try:
        results = store.similarity_search(q, k=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"retrieval error: {e}")

    contexts: List[str] = [t for t, _ in results]

    try:
        answer = call_llm(q, contexts)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    return {"answer": answer}