from pathlib import Path

from langchain_core.documents import Document

from rag_demo import embeddings
from rag_demo.embeddings import Embedding

from .fakes import FakeEmbeddingModel


def _documents():
    return [
        Document(page_content="alpha", metadata={"source": "a.pdf"}),
        Document(page_content="alpha", metadata={"source": "b.pdf"}),
        Document(page_content="beta", metadata={"source": "c.pdf"}),
    ]


def test_deduplicate_chunks_and_content_hash(tmp_path):
    embedder = Embedding(
        "doc.pdf",
        _documents(),
        embedding_model=FakeEmbeddingModel(),
        persist_root=tmp_path,
    )
    unique = embedder.deduplicate_chunks(_documents())
    assert len(unique) == 2
    assert embedder.content_hash() == embedder.content_hash()
    assert len(embedder.content_hash()) == 64


def test_collection_name_is_safe(tmp_path):
    embedder = Embedding(
        "doc.pdf",
        [],
        embedding_model=FakeEmbeddingModel(),
        persist_root=tmp_path,
    )
    name = embedder._collection_name("user with spaces", "doc/with/slashes")
    assert " " not in name
    assert "/" not in name
    assert len(name) <= 63


def test_vectorstore_persists_and_reloads(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "build_embedding_model", lambda *args, **kwargs: FakeEmbeddingModel())
    embedder = Embedding(
        "doc.pdf",
        _documents(),
        embedding_model=FakeEmbeddingModel(),
        persist_root=tmp_path,
    )
    result = embedder.create_or_load_vectorstore("user-1", "doc-1", knn=2)

    assert result["chunk_count"] == 2
    assert Path(result["persist_path"]).exists()
    assert result["collection_name"].startswith("rag_user-1_")
    assert result["retriever"].invoke("alpha")

    reloaded = Embedding.load_retriever(
        result["persist_path"],
        result["collection_name"],
        model_name="nomic-embed-text:latest",
        knn=2,
    )
    assert reloaded.invoke("alpha")
