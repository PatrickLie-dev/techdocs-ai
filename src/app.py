# src/app.py
# Flask REST API — wraps RAGPipeline into HTTP endpoints
# Run: python src/app.py

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.retriever import RAGPipeline, get_pipeline

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # allow cross-origin requests (useful for frontend integration)

# Initialize pipeline once at startup (expensive: loads model + connects to DB)
pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    """Return the global pipeline, raising 503 if not ready."""
    if pipeline is None:
        raise RuntimeError("Pipeline not initialized.")
    return pipeline


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/api/chat")
def chat():
    """
    POST /api/chat
    Body:  { "question": "What is Linux?" }
    Returns: { "answer": "...", "sources": [{ "source": "...", "page": ... }] }
    """
    body = request.get_json(silent=True)
    if not body or "question" not in body:
        return jsonify({"error": "Request body must contain a 'question' field."}), 400

    question: str = body["question"].strip()
    if not question:
        return jsonify({"error": "'question' cannot be empty."}), 400

    try:
        result = _get_pipeline().query(question)
        return jsonify({"answer": result["answer"], "sources": result["sources"]}), 200
    except RuntimeError as e:
        logger.error(f"Pipeline error: {e}")
        return jsonify({"error": "Service unavailable. Index may not be built yet."}), 503
    except Exception as e:
        logger.exception("Unexpected error in /api/chat")
        return jsonify({"error": "Internal server error."}), 500


@app.get("/api/health")
def health():
    """
    GET /api/health
    Returns: { "status": "ok", "indexed_chunks": 1234 }
    """
    try:
        p = _get_pipeline()
        chunk_count = p.vectorstore._collection.count()
        return jsonify({"status": "ok", "indexed_chunks": chunk_count}), 200
    except RuntimeError:
        return jsonify({"status": "unavailable", "indexed_chunks": 0}), 503


@app.get("/api/docs")
def list_docs():
    """
    GET /api/docs
    Returns: { "documents": ["intro-linux.pdf", ...] }
    """
    try:
        sources = _get_pipeline().list_indexed_sources()
        return jsonify({"documents": sources}), 200
    except RuntimeError:
        return jsonify({"error": "Service unavailable. Index may not be built yet."}), 503
    except Exception:
        logger.exception("Unexpected error in /api/docs")
        return jsonify({"error": "Internal server error."}), 500


# ---------------------------------------------------------------------------
# App startup
# ---------------------------------------------------------------------------

def initialize_pipeline():
    """Initialize pipeline — called at startup regardless of run mode."""
    global pipeline
    logger.info("Initializing RAG pipeline...")
    try:
        pipeline = get_pipeline()
        logger.info("Pipeline initialized successfully.")
    except ValueError as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        logger.error("Make sure you have run: python src/indexer.py first")
        # Tidak sys.exit() — biarkan app tetap jalan, health endpoint akan return 503

# Initialize saat module di-import (works dengan gunicorn)
initialize_pipeline()

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Starting Flask on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)


@app.get("/")
def index():
    return jsonify({
        "service": "TechDocs AI",
        "status": "running",
        "endpoints": {
            "health": "/api/health",
            "chat": "/api/chat (POST)",
            "docs": "/api/docs"
        }
    }), 200