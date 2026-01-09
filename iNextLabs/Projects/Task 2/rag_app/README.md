# README.md

# Retrieval-Augmented Generation (RAG) Application

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system that combines semantic search over a private knowledge base with large language model (LLM) generation. The system ensures responses are grounded in indexed source documents rather than relying solely on the LLM’s parametric knowledge.

The application follows a standard **RAG pipeline**:
1. Offline document indexing with embeddings and FAISS
2. Online query-time retrieval of relevant chunks
3. Context-aware answer generation via an LLM

---

## 1. System Architecture Overview

### High-Level Architecture

<img width="521" height="621" alt="Understanding RAG New drawio" src="https://github.com/user-attachments/assets/73473d50-ceb6-455a-9df4-7d0f4d44d2d3" />


### Component Interaction

- **Client** sends a query via HTTP or CLI
- **FastAPI** orchestrates the workflow
- **Retriever** converts query → embedding → FAISS search
- **LLM Module** receives query + retrieved context
- **Final Answer** is returned to the client

### Technology Stack

| Layer | Technology |
|------|-----------|
| API Server | FastAPI |
| Embeddings | OpenRouter (embedding model) |
| Vector Store | FAISS |
| LLM Inference | OpenRouter-compatible LLM |
| Runtime | Python 3.10+ |

---

## 2. Data Flow

### End-to-End Query Flow

1. **User Query Submission**
   - User submits a natural language question via `/chat`

2. **Query Embedding**
   - Query is converted into a dense vector using the same embedding model used at index time

3. **Vector Search (FAISS)**
   - FAISS performs similarity search against stored document embeddings
   - Top-k (default = 5) most similar chunks are returned

4. **Context Assembly**
   - Retrieved chunks are concatenated and structured as context

5. **LLM Generation**
   - Query + retrieved context is sent to the LLM
   - LLM generates an answer constrained by the provided context

6. **Response Delivery**
   - API returns the generated answer (optionally with metadata)

### Embeddings Usage

- Same embedding model is used for:
  - Document chunks (offline indexing)
  - User queries (online retrieval)
- This ensures vector space consistency

### FAISS Index Structure

- Index type: Flat or IVF (depending on configuration)
- Stored elements:
  - Vector embeddings
  - Chunk metadata (source text, chunk ID)
- Similarity metric: cosine similarity or inner product

---

## 3. Component Details

### Data Preparation

**Location:** `data/inextlabs.txt`

- Raw knowledge base text file
- Single or multi-document source
- Preprocessing performed during indexing

---

### Indexing (`build_index.py`)

**Responsibilities:**
- Load raw text
- Split text into overlapping chunks
- Generate embeddings for each chunk
- Build FAISS index
- Persist index and metadata to disk

**Key Steps:**
1. Text chunking
2. Embedding generation via OpenRouter
3. FAISS index creation
4. Index serialization

---

### Retrieval (`retrieval.py` / `app/faiss_store.py`)

#### `FaissStore`

**Purpose:**
- Load persisted FAISS index
- Execute similarity search

**Key Methods:**
- `load_index()` – Loads index and metadata
- `search(query_embedding, top_k)` – Returns relevant chunks

#### `Retriever`

**Purpose:**
- High-level retrieval abstraction
- Converts query → embedding → FAISS results

---

### LLM Interface (`app/llm.py`)

#### `call_llm()`

**Purpose:**
- Send prompt to LLM with retrieved context

**Inputs:**
- User query
- Retrieved document chunks

**Outputs:**
- Context-grounded natural language answer

---

### API Layer (`api.py`)

#### Endpoint: `/chat`

**Method:** `POST`

**Request Body:**
```json
{
  "query": "What does iNextLabs specialize in?"
}
```

**Response:**
```json
{
  "answer": "iNextLabs focuses on building AI agents for enterprise transformation."
}
```

---

## 4. Setup and Deployment

### Environment Setup

```bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

Set environment variables:
```bash
export OPENROUTER_API_KEY=your_api_key
```

---

### Build FAISS Index

```bash
python build_index.py
```

Artifacts generated:
- FAISS index file
- Metadata pickle/JSON

---

### Run API Server

```bash
uvicorn api:app --reload
```

Server runs at:
```
http://localhost:8000
```

---

### Command-Line Query (Optional)

```bash
python retrieval.py "What products does the company offer?"
```

---

## 5. Usage Examples

### API Example

**Request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain the inFlow platform."}'
```

**Response:**
```json
{
  "answer": "The inFlow platform is designed to build intelligent AI agents that can reason, plan, and act."
}
```

### Expected Behavior

- Answers are grounded in indexed documents
- If no relevant context is found, system may:
  - Return a fallback response
  - Indicate insufficient knowledge

### Error Handling

- Missing index → startup failure
- Invalid API key → LLM/embedding errors
- Empty query → validation error

---

## 6. Performance and Scalability Considerations

### Retrieval Parameters

- `top_k = 5` (configurable)
- Higher values increase recall but reduce response speed

### Index Size & Memory

- FAISS index resides in memory
- Scales linearly with number of chunks

### Optimization Opportunities

- Use IVF or HNSW FAISS indexes
- Implement hybrid search (BM25 + vectors)
- Add caching for frequent queries
- Batch embeddings during indexing
- Stream LLM responses

---

## Future Enhancements

- Multi-document support
- Source citations in responses
- Query rewriting for better recall
- Role-based system prompts
- Evaluation metrics (precision, recall)

---

## License

Internal / Educational Use

