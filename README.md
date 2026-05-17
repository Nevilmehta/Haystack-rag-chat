# AI Second Brain

A local, privacy-first RAG (Retrieval-Augmented Generation) chatbot that lets you upload your own documents (PDF/TXT) and chat with them using a local LLM via Ollama — no cloud, no data leaving your machine.

## Features

- **Document upload** — upload PDF and TXT files through the UI; they are chunked, embedded, and indexed instantly
- **Hybrid retrieval** — combines BM25 (keyword) + semantic (embedding) search, then re-ranks results with a cross-encoder
- **Streaming responses** — answers stream token-by-token via Server-Sent Events
- **Conversational memory** — follows up on pronouns and references across turns ("he", "that project", etc.)
- **Response caching** — identical questions return instantly from an in-memory TTL cache (1 hour)
- **Explainability panel** — shows retrieved source chunks, relevance scores, and a confidence metric alongside every answer
- **Prompt injection protection** — blocks known jailbreak patterns before they reach the LLM
- **API key auth** — all endpoints protected via `X-API-Key` header
- **Docker support** — single-command backend deployment

## Architecture

```
User (Streamlit UI)
       │
       ▼
  FastAPI Backend
       │
  ┌────┴────────────────────────────────────┐
  │           Haystack RAG Pipeline          │
  │                                          │
  │  Query ──► BM25 Retriever ──┐            │
  │       └──► Embedding        ├──► Joiner  │
  │            Retriever ───────┘     │      │
  │                                   ▼      │
  │                          Cross-Encoder   │
  │                            Ranker        │
  │                               │          │
  │                          PromptBuilder   │
  │                               │          │
  │                        OllamaGenerator   │
  └───────────────────────────────┼──────────┘
                                  │
                            Local Ollama
                          (llama3.2:1b)
```

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI + Uvicorn |
| RAG Framework | Haystack AI |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Ollama (`llama3.2:1b`) |
| PDF parsing | pypdf |
| Containerization | Docker + Docker Compose |

## Project Structure

```
AI Second Brain/
├── backend/
│   ├── main.py          # FastAPI app, route definitions
│   ├── rag_pipeline.py  # Haystack pipeline, RAGService class
│   ├── config.py        # Environment config and constants
│   ├── schemas.py       # Pydantic request/response models
│   ├── security.py      # API key auth, prompt injection guard
│   ├── cache.py         # In-memory TTL cache
│   └── utils.py         # PDF reader, text chunker, confidence scorer
├── frontend/
│   └── app.py           # Streamlit 3-panel chat UI
├── data/                # Uploaded documents stored here
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running locally
- The model pulled: `ollama pull llama3.2:1b`

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd "AI Second Brain"
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root:

```env
API_KEY=your-secret-key
OLLAMA_URL=http://localhost:11434
```

### 3. Start the backend

```bash
uvicorn backend.main:app --reload
```

### 4. Start the frontend

```bash
streamlit run frontend/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Docker (backend only)

```bash
docker-compose up --build
```

The backend runs on port `8000`. Ollama must be running on the host machine — the container connects via `host.docker.internal:11434`.

## API Endpoints

All endpoints require the `X-API-Key` header.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Single-turn Q&A (full response) |
| `POST` | `/chat/stream` | Streaming Q&A (SSE, token-by-token) |
| `POST` | `/upload` | Upload and index a PDF or TXT file |

### Example request

```bash
curl -X POST http://localhost:8000/chat \
  -H "X-API-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "Who is Nevil Mehta?"}'
```

### Response shape

```json
{
  "answer": "...",
  "sources": [
    {
      "source": "resume.pdf",
      "chunk_id": 3,
      "score": 0.91,
      "content": "..."
    }
  ],
  "confidence": 0.87,
  "response_time_seconds": 2.34,
  "cached": false
}
```

## RAG Pipeline Details

**Chunking** — documents are split into 350-character chunks with 100-character overlap to preserve context across boundaries.

**Hybrid retrieval** — BM25 handles exact keyword matches; embedding retrieval handles semantic similarity. Both retrievers fetch `top_k=3` documents each and results are merged before re-ranking.

**Re-ranking** — a cross-encoder (`ms-marco-MiniLM-L-6-v2`) scores each candidate chunk against the query and returns the top 2 most relevant.

**Confidence score** — the average cross-encoder score of the re-ranked documents, clamped to `[0.0, 1.0]`.

**Caching** — questions are normalized (lowercased + stripped) and hashed with SHA-256. Cache TTL is 1 hour; cleared automatically on new document upload.

**Conversational memory** — the full chat history is injected into the prompt so the LLM can resolve references across turns.

## Security

- All API routes require a valid `X-API-Key` header
- Incoming questions are validated for length (max 1000 chars) and scanned for prompt injection patterns before reaching the pipeline
- Blocked patterns include: `ignore previous instructions`, `jailbreak`, `reveal system prompt`, and others
