import hashlib
import inspect
import logging
import os
import re
import shutil
from pathlib import Path
from typing import List, Optional, Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_CHROMA_DIR = "./chroma"


def _safe_segment(value: str, max_length: int = 60) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]", "_", str(value))
    value = re.sub(r"^[^a-zA-Z0-9]+", "", value)
    value = re.sub(r"[^a-zA-Z0-9]+$", "", value)
    value = value[:max_length].rstrip("._-")
    return value or "doc"


def _sha256_texts(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def build_embedding_model(model_type: str, model_name: str, **kwargs):
    model_type = model_type.lower()
    if model_type == "ollama":
        kwargs.setdefault("base_url", os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL))
        return OllamaEmbeddings(model=model_name or DEFAULT_EMBEDDING_MODEL, **kwargs)
    if model_type == "huggingface":
        model_kwargs = {
            "cache_folder": "./RAG",
            "model_kwargs": {"device": "cpu"},
            "encode_kwargs": {"normalize_embeddings": False},
        }
        signature = inspect.signature(HuggingFaceEmbeddings).parameters
        if "model_name" in signature:
            model_kwargs["model_name"] = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        else:
            model_kwargs["model"] = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        return HuggingFaceEmbeddings(**model_kwargs)
    raise ValueError(
        f"Unsupported embedding model type: {model_type}. Supported types are ollama and huggingface."
    )


class BatchedEmbeddingWrapper:
    def __init__(self, embedder, batch_size: int = 32):
        self.embedder = embedder
        self.batch_size = batch_size

    def embed_query(self, text: str):
        return self.embedder.embed_query(text)

    def embed_documents(self, texts: List[str]):
        all_embeddings = []
        for index in range(0, len(texts), self.batch_size):
            batch = [str(text) for text in texts[index : index + self.batch_size]]
            all_embeddings.extend(self.embedder.embed_documents(batch))
        return all_embeddings


class Embedding:
    def __init__(
        self,
        path: str,
        chunks: List[Document],
        *,
        embedding_model=None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        persist_root: Optional[Path] = None,
        batch_size: int = 32,
    ):
        self.path = path
        self.chunks = chunks
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        self.persist_root = Path(persist_root or os.getenv("CHROMA_DIR", DEFAULT_CHROMA_DIR))
        self.batch_size = batch_size
        self.embeddingModel = embedding_model or build_embedding_model(
            "ollama",
            self.model_name,
            base_url=self.base_url,
        )

    def set_embedding_model(self, model_type: str, model_name: str, **kwargs):
        try:
            model_type = model_type.lower()
            if model_type == "ollama":
                self.model_name = model_name or DEFAULT_EMBEDDING_MODEL
                kwargs.setdefault("base_url", self.base_url)
            self.embeddingModel = build_embedding_model(model_type, model_name, **kwargs)
            test_embedding = self.embeddingModel.embed_query("Test Document")
            logger.info(
                "Switched to %s embedding model, dimension: %s",
                model_type,
                len(test_embedding),
            )
        except Exception as error:
            logger.error("Failed to set embedding model: %s", error)
            raise

    def _ensure_model(self):
        if self.embeddingModel is None:
            raise RuntimeError("Embedding model not initialized")

    def compute_embedding(self, text: str):
        self._ensure_model()
        return self.embeddingModel.embed_query(text)

    def embed_document_in_batches(self, documents, batch_size: Optional[int] = None):
        self._ensure_model()
        size = batch_size or self.batch_size
        all_embeddings = []
        for index in range(0, len(documents), size):
            batch = [str(document) for document in documents[index : index + size]]
            all_embeddings.extend(self.embeddingModel.embed_documents(batch))
        return all_embeddings

    def deduplicate_chunks(self, documents: List[Document]) -> List[Document]:
        seen = set()
        unique: List[Document] = []
        for document in documents:
            normalized = re.sub(r"\s+", " ", document.page_content).strip()
            key = hashlib.md5(normalized.encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            unique.append(document)
        if len(unique) < len(documents):
            logger.info("Deduplicated chunks: %s -> %s", len(documents), len(unique))
        return unique

    def content_hash(self) -> str:
        normalized = [
            re.sub(r"\s+", " ", document.page_content).strip()
            for document in self.chunks
        ]
        return _sha256_texts(normalized)

    def _collection_name(self, user_id: Optional[str] = None, document_id: Optional[str] = None) -> str:
        user_part = _safe_segment(user_id or "anonymous", 20)
        document_part = _safe_segment(document_id or Path(self.path).stem, 32)
        name = f"rag_{user_part}_{document_part}_v"[:63].rstrip("._-")
        return name or "rag_default_v"

    def _persist_dir(self, user_id: Optional[str] = None, document_id: Optional[str] = None) -> Path:
        user_part = _safe_segment(user_id or "anonymous", 40)
        document_part = _safe_segment(document_id or Path(self.path).stem, 60)
        return self.persist_root / user_part / document_part

    def create_or_load_vectorstore(
        self,
        userId: Optional[str],
        documentId: Optional[str] = None,
        saveVector: bool = True,
        knn: int = 5,
        **kwargs,
    ):
        self._ensure_model()
        persist_dir = self._persist_dir(userId, documentId)
        persist_dir.mkdir(parents=True, exist_ok=True)
        collection_name = self._collection_name(userId, documentId)
        embedding_function = BatchedEmbeddingWrapper(self.embeddingModel, self.batch_size)

        try:
            vector_store = Chroma(
                persist_directory=str(persist_dir),
                collection_name=collection_name,
                embedding_function=embedding_function,
            )

            if vector_store._collection.count() == 0:
                documents = self.deduplicate_chunks(self.chunks)
                if not documents:
                    raise ValueError("No documents available to index")
                vector_store.add_documents(documents)

            retriever = vector_store.as_retriever(search_kwargs={"k": knn}, **kwargs)
            chunk_count = vector_store._collection.count()
            logger.info(
                "Vector store ready at %s with %s chunks in collection %s",
                persist_dir,
                chunk_count,
                collection_name,
            )
            return {
                "retriever": retriever,
                "collection_name": collection_name,
                "persist_path": str(persist_dir),
                "chunk_count": chunk_count,
                "content_hash": self.content_hash(),
            }
        except Exception as error:
            logger.error("Error in vector store creation or loading: %s", error)
            raise

    @classmethod
    def load_retriever(
        cls,
        persist_path: str,
        collection_name: str,
        *,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        knn: int = 5,
        **kwargs,
    ):
        embedding_model = build_embedding_model(
            "ollama",
            model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            base_url=base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        )
        vector_store = Chroma(
            persist_directory=str(persist_path),
            collection_name=collection_name,
            embedding_function=BatchedEmbeddingWrapper(embedding_model),
        )
        if vector_store._collection.count() == 0:
            raise ValueError(f"Collection {collection_name} is empty or missing")
        return vector_store.as_retriever(search_kwargs={"k": knn}, **kwargs)

    def cleanup_vectorstore(
        self,
        user_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ):
        persist_dir = self._persist_dir(user_id, document_id)
        if persist_dir.exists():
            shutil.rmtree(persist_dir)
            logger.info("Vector store at %s cleaned up", persist_dir)
        else:
            logger.info("No vector store found at %s", persist_dir)
