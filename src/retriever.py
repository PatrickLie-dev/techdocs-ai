# src/retriever.py
# Responsible: load ChromaDB → retrieve relevant chunks → call Groq LLM → return answer + sources
# Run standalone: python src/retriever.py

import logging
import os
import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COLLECTION_NAME = "techdocs"

PROMPT_TEMPLATE = """You are a helpful technical documentation assistant.
Use the context below to answer the question as completely as possible.
If the context contains relevant information, synthesize it into a clear answer even if it is not stated word-for-word.
Only say "I don't have enough information in the documentation to answer that" if the context is genuinely unrelated to the question.

Context:
{context}

Question: {question}

Answer:"""


class RAGResponse(TypedDict):
    answer: str
    sources: list[dict]


class RAGPipeline:
    """
    End-to-end RAG pipeline: query → retrieve chunks → generate answer via Groq.
    """

    def __init__(
        self,
        persist_dir: str,
        embedding_model: str,
        groq_model: str,
        top_k: int,
    ) -> None:
        self.persist_dir = persist_dir
        self.top_k = top_k

        logger.info(f"Loading embeddings: {embedding_model}")
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info(f"Connecting to ChromaDB at: {persist_dir}")
        chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            client=chroma_client,
        )

        doc_count = self.vectorstore._collection.count()
        if doc_count == 0:
            raise ValueError(
                f"ChromaDB at '{persist_dir}' is empty. Run indexer.py first."
            )
        logger.info(f"ChromaDB loaded: {doc_count} chunks available.")

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

        logger.info(f"Loading Groq model: {groq_model}")
        self.llm = ChatGroq(
            model=groq_model,
            temperature=0.1,       # low temp for factual Q&A
            max_tokens=1024,
        )

        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=PROMPT_TEMPLATE,
        )

        # Build LCEL chain
        self._chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        logger.info("RAG pipeline ready.")

    @staticmethod
    def _format_docs(docs) -> str:
        """Concatenate retrieved chunks into a single context string."""
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def query(self, question: str) -> RAGResponse:
        """
        Run a question through the full RAG pipeline.

        Args:
            question: User's question string.

        Returns:
            RAGResponse with 'answer' (str) and 'sources' (list of dicts).
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        logger.info(f"Query: {question!r}")

        # Retrieve relevant chunks (separate call to capture metadata)
        retrieved_docs = self.retriever.invoke(question)

        # Generate answer
        answer = self._chain.invoke(question)

        # Build deduplicated source list
        seen: set[str] = set()
        sources: list[dict] = []
        for doc in retrieved_docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", None)
            key = f"{source}:{page}"
            if key not in seen:
                seen.add(key)
                entry = {"source": Path(source).name, "page": page}
                sources.append(entry)

        logger.info(f"Answer generated. Sources: {[s['source'] for s in sources]}")
        return RAGResponse(answer=answer, sources=sources)

    def list_indexed_sources(self) -> list[str]:
        """Return unique document filenames currently in the vector store."""
        results = self.vectorstore.get(include=["metadatas"])
        sources: set[str] = set()
        for meta in results.get("metadatas", []):
            if meta and "source" in meta:
                sources.add(Path(meta["source"]).name)
        return sorted(sources)


def get_pipeline() -> RAGPipeline:
    """Factory: build RAGPipeline from .env config. Call once at app startup."""
    return RAGPipeline(
        persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        top_k=int(os.getenv("TOP_K_RESULTS", "4")),
    )


if __name__ == "__main__":
    questions = [
        "What is Linux?",
        "How do I list files in a directory?",
        "What is the difference between a process and a thread?",
    ]

    try:
        pipeline = get_pipeline()
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = pipeline.query(q)
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
