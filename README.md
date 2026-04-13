# TechDocs AI 🤖

A production-ready RAG (Retrieval-Augmented Generation) chatbot that answers questions from technical IT documentation — powered by LangChain, ChromaDB, and Groq.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green?logo=chainlink)
![Flask](https://img.shields.io/badge/Flask-3.1-black?logo=flask)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.6-orange)
![Groq](https://img.shields.io/badge/Groq-LLaMA3.3--70B-purple)

---

## What it does

Upload any technical PDF documentation and ask questions in natural language. The system retrieves the most relevant chunks from the document and generates accurate, grounded answers — with source page references.

```
User: "How do I list files in a directory?"

TechDocs AI: "To list files in a directory, use the `ls` command..."
             Sources: intro-linux.pdf (page 61, 28, 67)
```

No hallucination — answers are strictly grounded in your documents.

---

## Architecture

```
Documents (PDF/TXT/MD)
        ↓
  [Indexing Pipeline]
  Text Splitter (chunk_size=500, overlap=50)
        ↓
  HuggingFace Embeddings (all-MiniLM-L6-v2)
        ↓
  ChromaDB (vector store, persisted locally)

        ↓ (query time)

  [Query Pipeline]
  User Question → Embed → Similarity Search
        ↓
  Top-4 Relevant Chunks
        ↓
  Prompt Builder (context + question)
        ↓
  Groq LLM (llama-3.3-70b-versatile)
        ↓
  {"answer": "...", "sources": [...]}
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangChain 0.3 |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local, free) |
| Vector Store | ChromaDB |
| LLM | Groq API — `llama-3.3-70b-versatile` |
| API | Flask REST API |
| Document Loaders | PyPDF, TextLoader (LangChain Community) |

---

## Project Structure

```
techdocs-ai/
├── src/
│   ├── loader.py       # Load PDF/TXT/MD → Document objects
│   ├── indexer.py      # Text splitting + embedding + ChromaDB
│   ├── retriever.py    # RAGPipeline class: query → answer + sources
│   └── app.py          # Flask REST API
├── documents/          # Place your PDF/TXT/MD files here
├── chroma_db/          # Auto-generated vector store (gitignored)
├── .env.example
└── requirements.txt
```

---

## Getting Started

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/techdocs-ai.git
cd techdocs-ai

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here     # Free at console.groq.com
GROQ_MODEL=llama-3.3-70b-versatile
CHROMA_PERSIST_DIR=./chroma_db
DOCUMENTS_DIR=./documents
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RESULTS=4
```

### 3. Add your documents

Place any `.pdf`, `.txt`, or `.md` files into the `documents/` folder.

### 4. Build the index

```bash
python src/indexer.py
```

First run downloads the embedding model (~80MB) and indexes all documents.
Subsequent runs are skipped unless you pass `--force`.

```bash
python src/indexer.py --force    # Re-index from scratch
```

### 5. Start the API

```bash
python src/app.py
```

---

## API Reference

### `GET /api/health`

Returns server status and number of indexed chunks.

```bash
curl http://localhost:5000/api/health
```

```json
{
  "status": "ok",
  "indexed_chunks": 1280
}
```

---

### `GET /api/docs`

Returns list of indexed documents.

```bash
curl http://localhost:5000/api/docs
```

```json
{
  "documents": ["intro-linux.pdf"]
}
```

---

### `POST /api/chat`

Ask a question. Returns answer and source references.

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the Linux kernel?"}'
```

```json
{
  "answer": "The Linux kernel is the core component of the Linux operating system...",
  "sources": [
    {"source": "intro-linux.pdf", "page": 16},
    {"source": "intro-linux.pdf", "page": 18}
  ]
}
```

---

## Performance

Tested on `intro-linux.pdf` (223 pages, 1280 chunks):

| Metric | Result |
|---|---|
| Index build time | ~45 seconds (first run) |
| Query response time | < 2 seconds |
| Embedding model | Local (no API cost) |
| LLM cost | Free (Groq free tier) |

---

## Roadmap

- [x] Core RAG pipeline (loader → indexer → retriever)
- [x] Flask REST API with health check and docs endpoint
- [ ] Multi-document upload via API endpoint
- [ ] Dockerize for cloud deployment
- [ ] MLflow experiment tracking
- [ ] Chat history / conversation memory

---

## Author

**Patrick Lie**
BSc Information Technology (IoT Specialization)
Asia Pacific University, Jakarta — April 2026

[LinkedIn](https://linkedin.com/in/patrick-lie-315964302/) · [GitHub](https://github.com/PatrickLie-dev)