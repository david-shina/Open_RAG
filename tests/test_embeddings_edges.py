import pytest

from rag_demo import embeddings
from rag_demo.embeddings import Embedding

from .fakes import FakeEmbeddingModel


def test_build_embedding_model_rejects_unknown_type():
    with pytest.raises(ValueError):
        embeddings.build_embedding_model("unknown", "model")


def test_build_ollama_embedding_model_uses_base_url():
    model = embeddings.build_embedding_model(
        "ollama",
        "nomic-embed-text:latest",
        base_url="http://localhost:11434",
    )
    assert model.model == "nomic-embed-text:latest"


def test_cleanup_vectorstore_removes_directory(tmp_path):
    embedder = Embedding(
        "doc.pdf",
        [],
        embedding_model=FakeEmbeddingModel(),
        persist_root=tmp_path,
    )
    persist_dir = embedder._persist_dir("user-1", "doc-1")
    persist_dir.mkdir(parents=True)
    assert persist_dir.exists()

    embedder.cleanup_vectorstore("user-1", "doc-1")
    assert not persist_dir.exists()


def test_empty_vectorstore_raises(tmp_path):
    embedder = Embedding(
        "doc.pdf",
        [],
        embedding_model=FakeEmbeddingModel(),
        persist_root=tmp_path,
    )
    with pytest.raises(ValueError):
        embedder.create_or_load_vectorstore("user-1", "empty-doc")


def test_set_embedding_model(monkeypatch, tmp_path):
    embedder = Embedding(
        "doc.pdf",
        [],
        embedding_model=FakeEmbeddingModel(),
        persist_root=tmp_path,
    )
    monkeypatch.setattr(embeddings, "build_embedding_model", lambda *args, **kwargs: FakeEmbeddingModel())

    embedder.set_embedding_model("ollama", "nomic-embed-text:latest")
    assert embedder.compute_embedding("hello") == [0.1, 0.2, 0.3]

def test_load_retriever_rejects_empty_collection(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "build_embedding_model", lambda *args, **kwargs: FakeEmbeddingModel())
    with pytest.raises(ValueError):
        Embedding.load_retriever(str(tmp_path / "empty"), "rag_empty_v")
