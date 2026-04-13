# src/indexer.py
# Responsible: split documents into chunks → embed → persist to ChromaDB
# Run standalone: python src/indexer.py [--force]

import argparse
import logging
import os
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.loader import load_documents

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COLLECTION_NAME = "techdocs"


def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Load HuggingFace embedding model (runs locally, no API key needed)."""
    logger.info(f"Loading embedding model: {model_name}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def is_already_indexed(persist_dir: str) -> bool:
    """Return True if ChromaDB already has documents in the collection."""
    db_path = Path(persist_dir)
    if not db_path.exists():
        return False
    try:
        client = chromadb.PersistentClient(
            path=str(db_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        collections = client.list_collections()
        for col in collections:
            if col.name == COLLECTION_NAME:
                count = client.get_collection(COLLECTION_NAME).count()
                if count > 0:
                    logger.info(f"Found existing index: {count} chunks in '{COLLECTION_NAME}'.")
                    return True
    except Exception:
        pass
    return False


def build_index(
    docs_dir: str,
    persist_dir: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    force: bool = False,
) -> int:
    """
    Load documents → split → embed → persist to ChromaDB.

    Args:
        docs_dir:        Path to documents folder.
        persist_dir:     Path to ChromaDB persistence directory.
        embedding_model: HuggingFace model name for embeddings.
        chunk_size:      Max tokens per chunk.
        chunk_overlap:   Overlap between consecutive chunks.
        force:           If True, re-index even if DB already exists.

    Returns:
        Number of chunks indexed.

    Raises:
        FileNotFoundError: If docs_dir doesn't exist.
        ValueError:        If no documents found.
    """
    if not force and is_already_indexed(persist_dir):
        logger.info("Skipping indexing — use --force to re-index.")
        return 0

    # 1. Load raw documents
    logger.info(f"Loading documents from: {docs_dir}")
    documents = load_documents(docs_dir)
    logger.info(f"Loaded {len(documents)} pages/documents.")

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap}).")

    # 3. Load embedding model
    embeddings = get_embeddings(embedding_model)

    # 4. Persist to ChromaDB (overwrites existing collection when force=True)
    persist_path = str(Path(persist_dir))
    logger.info(f"Persisting to ChromaDB at: {persist_path}")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_path,
        client_settings=ChromaSettings(anonymized_telemetry=False),
    )

    total = vectorstore._collection.count()
    logger.info(f"Indexing complete. Total chunks in DB: {total}")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ChromaDB index from documents.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing even if ChromaDB already exists.",
    )
    args = parser.parse_args()

    docs_dir = os.getenv("DOCUMENTS_DIR", "./documents")
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))

    try:
        count = build_index(
            docs_dir=docs_dir,
            persist_dir=persist_dir,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            force=args.force,
        )
        if count > 0:
            print(f"\nDone. {count} chunks indexed to '{persist_dir}'.")
        else:
            print("\nIndex already up-to-date. Run with --force to re-index.")
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)
