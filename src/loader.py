# src/loader.py
# Bertanggung jawab: load dokumen dari folder → list of Document objects
# Mirip cara kamu load data sensor di Freshly, tapi inputnya file bukan Firebase

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_documents(docs_dir: str = "./documents") -> List[Document]:
    """
    Load semua dokumen dari direktori.
    Mendukung: PDF, TXT, MD

    Returns:
        List of Document objects dengan page_content dan metadata.

    Raises:
        FileNotFoundError: Kalau direktori tidak ada.
        ValueError: Kalau tidak ada dokumen ditemukan.
    """
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        raise FileNotFoundError(f"Directory '{docs_dir}' tidak ditemukan.")

    all_docs: List[Document] = []

    # --- Load PDF ---
    pdf_loader = DirectoryLoader(
        path=docs_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
        silent_errors=True,     # skip file yang corrupt, jangan crash
    )
    pdf_docs = pdf_loader.load()
    logger.info(f"PDF loaded: {len(pdf_docs)} halaman")
    all_docs.extend(pdf_docs)

    # --- Load TXT ---
    txt_loader = DirectoryLoader(
        path=docs_dir,
        glob="**/*.txt",
        loader_cls=lambda p: TextLoader(p, encoding="utf-8"),
        show_progress=True,
        silent_errors=True,
    )
    txt_docs = txt_loader.load()
    logger.info(f"TXT loaded: {len(txt_docs)} file")
    all_docs.extend(txt_docs)

    # --- Load MD ---
    md_loader = DirectoryLoader(
        path=docs_dir,
        glob="**/*.md",
        loader_cls=lambda p: TextLoader(p, encoding="utf-8"),
        show_progress=True,
        silent_errors=True,
    )
    md_docs = md_loader.load()
    logger.info(f"MD loaded: {len(md_docs)} file")
    all_docs.extend(md_docs)

    if not all_docs:
        raise ValueError(
            f"Tidak ada dokumen ditemukan di '{docs_dir}'. "
            "Tambahkan file PDF, TXT, atau MD ke folder tersebut."
        )

    logger.info(f"Total dokumen ter-load: {len(all_docs)}")
    return all_docs


def preview_documents(docs: List[Document], n: int = 3) -> None:
    """Helper untuk inspect dokumen setelah loading."""
    print(f"\n{'='*50}")
    print(f"Total dokumen: {len(docs)}")
    print(f"Preview {min(n, len(docs))} dokumen pertama:")
    print("="*50)

    for i, doc in enumerate(docs[:n]):
        print(f"\n[Doc {i+1}]")
        print(f"Source : {doc.metadata.get('source', 'unknown')}")
        print(f"Page   : {doc.metadata.get('page', '-')}")
        print(f"Length : {len(doc.page_content)} chars")
        print(f"Preview: {doc.page_content[:150].strip()}...")


# Quick test — jalankan langsung: python src/loader.py
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    docs_dir = os.getenv("DOCUMENTS_DIR", "./documents")

    try:
        documents = load_documents(docs_dir)
        preview_documents(documents)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error: {e}")