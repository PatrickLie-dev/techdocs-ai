# TechDocs AI

A production-deployed RAG (Retrieval-Augmented Generation) chatbot that answers questions from technical IT documentation — with source citations, evaluated via MLflow.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![Flask](https://img.shields.io/badge/Flask-3.1-black?logo=flask)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector--store-orange)
![Groq](https://img.shields.io/badge/Groq-LLaMA--3.3--70B-purple)
![MLflow](https://img.shields.io/badge/MLflow-evaluation-blue?logo=mlflow)
![Railway](https://img.shields.io/badge/Deployed-Railway-0B0D0E?logo=railway)

**Live demo:** https://techdocs-ai-production.up.railway.app

---

## Overview

TechDocs AI lets you drop any technical PDF documentation into a folder and immediately query it via a REST API in natural language. The system retrieves the most relevant passages and generates a grounded answer — with exact page references, so every claim is traceable.

```
POST /api/chat
{ "question": "How do I list files in a directory?" }

→ {
    "answer": "Use the `ls` command. For detailed output including hidden files, run `ls -la`...",
    "sources": [
      { "source": "intro-linux.pdf", "page": 61 },
      { "source": "intro-linux.pdf", "page": 28 }
    ]
  }
```

---

## Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| Orchestration | LangChain 0.3 (LCEL) | RAG chain composition |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace) | Local, CPU, free |
| Vector Store | ChromaDB | Persisted locally |
| LLM | Groq API — `llama-3.3-70b-versatile` | Free tier, ~1.4s/query |
| API | Flask 3.1 + Gunicorn | REST, CORS-enabled |
| Containerization | Docker (CPU-only torch) | Optimized image size |
| Deployment | Railway | Auto-deploy from GitHub |
| Evaluation | MLflow | Latency, relevance, source coverage |

---

## Architecture

```
                    ┌─────────────────────────────────┐
                    │         INDEXING (offline)       │
                    │                                   │
  documents/        │  loader.py → indexer.py           │
  (PDF/TXT/MD)  ──► │  RecursiveCharacterTextSplitter   │
                    │  chunk_size=500, overlap=50        │
                    │         ↓                         │
                    │  HuggingFace Embeddings (CPU)      │
                    │         ↓                         │
                    │  ChromaDB (persisted)              │
                    └─────────────────────────────────┘

                    ┌─────────────────────────────────┐
                    │         QUERY (runtime)          │
                    │                                   │
  User Question ──► │  Embed question                   │
                    │  Similarity search → top-4 chunks │
                    │  Build prompt (context + question) │
                    │  Groq LLM (llama-3.3-70b)         │
                    │         ↓                         │
                    │  { answer, sources }               │
                    └─────────────────────────────────┘
```

---

## Project Structure

```
techdocs-ai/
├── src/
│   ├── loader.py            # Load PDF/TXT/MD → LangChain Documents
│   ├── indexer.py           # Chunk → embed → store in ChromaDB
│   ├── retriever.py         # RAGPipeline class: query → answer + sources
│   └── app.py               # Flask REST API (3 endpoints)
├── mlflow_portfolio/
│   └── rag_evaluator.py     # MLflow evaluation pipeline (5 query eval suite)
├── documents/               # Drop your docs here
├── chroma_db/               # Vector store (auto-generated)
├── Dockerfile
├── .env.example
└── requirements.txt
```

---

## API Reference

### `GET /api/health`
```json
{ "status": "ok", "indexed_chunks": 1280 }
```

### `GET /api/docs`
```json
{ "documents": ["intro-linux.pdf"] }
```

### `POST /api/chat`
Request:
```json
{ "question": "What is the Linux kernel?" }
```
Response:
```json
{
  "answer": "The Linux kernel is the core component...",
  "sources": [
    { "source": "intro-linux.pdf", "page": 16 },
    { "source": "intro-linux.pdf", "page": 18 }
  ]
}
```

---

## Running Locally

**Prerequisites:** Python 3.11+, a free [Groq API key](https://console.groq.com)

```bash
# 1. Clone and install
git clone https://github.com/PatrickLie-dev/techdocs-ai.git
cd techdocs-ai
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env → set GROQ_API_KEY

# 3. Add documents
# Place .pdf / .txt / .md files into documents/

# 4. Build the index
python src/indexer.py
# First run downloads embedding model (~80MB). Use --force to re-index.

# 5. Start the API
python src/app.py
# → http://localhost:5000
```

### Using Docker
```bash
docker build -t techdocs-ai .
docker run -p 5000:5000 --env-file .env techdocs-ai
```

---

## MLflow Evaluation

An automated evaluation suite (`mlflow_portfolio/rag_evaluator.py`) hits the deployed API with 5 queries across different categories and logs metrics to MLflow.

```bash
cd mlflow_portfolio
python rag_evaluator.py          # runs eval against Railway deployment
python -m mlflow ui --port 5001  # view results at http://127.0.0.1:5001
```

**Baseline results (intro-linux.pdf, 1280 chunks):**

| Metric | Value |
|---|---|
| Success rate | 100% |
| Avg latency | 1,424 ms |
| Avg relevance score | 0.66 |
| Avg sources returned | 3.4 |
| Failed queries | 0 |

---

## Screenshots

> _Add screenshots here after running the app._

**MLflow Evaluation Dashboard**
<!-- ![MLflow UI](screenshots/mlflow-dashboard.png) -->

**API Response — `/api/chat`**
<!-- ![API Chat Response](screenshots/api-chat.png) -->

**Deployment on Railway**
<!-- ![Railway Deployment](screenshots/railway-deploy.png) -->

---

## Roadmap

- [x] Core RAG pipeline (loader → indexer → retriever)
- [x] Flask REST API with health, docs, and chat endpoints
- [x] Dockerized for cloud deployment
- [x] Deployed on Railway
- [x] MLflow evaluation pipeline
- [ ] Multi-document upload via API
- [ ] Conversation memory / chat history
- [ ] Frontend UI

---

## Author

**Patrick Lie**
Fresh Graduate — Information Technology (IoT Specialization)
Asia Pacific University, Jakarta

[LinkedIn](https://linkedin.com/in/patrick-lie-315964302) · [GitHub](https://github.com/PatrickLie-dev)
