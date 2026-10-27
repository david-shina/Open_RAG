import uuid

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APIClient

from my_rag_api import views
from my_rag_api.models import DocumentRecord, QueryLog

from .fakes import FakeRetriever


class FakeEmbedding:
    last_chunks = None

    def __init__(self, path, chunks, **kwargs):
        self.path = path
        self.chunks = chunks
        FakeEmbedding.last_chunks = chunks

    def set_embedding_model(self, **kwargs):
        return None

    def create_or_load_vectorstore(self, userId, documentId=None, knn=5, **kwargs):
        return {
            "retriever": FakeRetriever(),
            "collection_name": f"rag_{userId}_{documentId}_v",
            "persist_path": f"/tmp/{userId}/{documentId}",
            "chunk_count": len(self.chunks),
        }

    def cleanup_vectorstore(self, user_id=None, document_id=None):
        return None


@pytest.mark.django_db
def test_upload_document_creates_record(tmp_path, monkeypatch):
    monkeypatch.setattr(views, "Embedding", FakeEmbedding)
    monkeypatch.setattr(
        views,
        "loadingDocument",
        lambda path: [
            type("Doc", (), {"page_content": "Example content", "metadata": {"source": "sample.pdf"}})()
        ],
    )
    monkeypatch.setattr(views.settings, "RAG_UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(views.settings, "RAG_CHROMA_DIR", tmp_path / "chroma")

    upload = SimpleUploadedFile("sample.pdf", b"%PDF-1.4\nbody\n%%EOF", content_type="application/pdf")
    response = APIClient().post(
        "/api/upload/",
        {"file": upload, "user_id": "user-1"},
        format="multipart",
    )

    assert response.status_code == 201, response.content
    payload = response.json()
    assert payload["success"] is True
    assert len(payload["document_id"]) == 32
    assert payload["chunks_created"] == 1
    assert DocumentRecord.objects.count() == 1
    record = DocumentRecord.objects.get()
    assert record.user_id == "user-1"
    assert record.original_name == "sample.pdf"
    assert record.collection_name == f"rag_user-1_{record.document_id}_v"
    assert FakeEmbedding.last_chunks[0].metadata["source"] == "sample.pdf"


@pytest.mark.django_db
def test_upload_rejects_invalid_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(views.settings, "RAG_UPLOAD_DIR", tmp_path / "uploads")
    upload = SimpleUploadedFile("sample.pdf", b"not a pdf", content_type="application/pdf")
    response = APIClient().post(
        "/api/upload/",
        {"file": upload, "user_id": "user-1"},
        format="multipart",
    )

    assert response.status_code == 400
    assert DocumentRecord.objects.count() == 0


@pytest.mark.django_db
def test_query_document_returns_citations_and_logs(monkeypatch):
    record = DocumentRecord.objects.create(
        user_id="user-1",
        document_id=uuid.uuid4().hex,
        original_name="sample.pdf",
        stored_path="/tmp/sample.pdf",
        collection_name="rag_user-1_doc_v",
        persist_path="/tmp/chroma",
        chunk_count=1,
        content_hash="a" * 64,
        embedding_model="nomic-embed-text:latest",
    )
    document = type(
        "Doc",
        (),
        {
            "page_content": "Paris is the capital of France.",
            "metadata": {"source": "sample.pdf", "page": 1, "citation_id": "S1"},
        },
    )()
    source = {
        "citation_id": "S1",
        "source": "sample.pdf",
        "page": 1,
        "chunk_index": 0,
        "url": None,
        "snippet": "Paris is the capital of France.",
    }
    monkeypatch.setattr(views, "_get_retriever", lambda record, top_k: FakeRetriever([document]))
    monkeypatch.setattr(
        views,
        "run_rag_agent",
        lambda *args, **kwargs: {
            "generation": "Paris is the capital of France [S1].",
            "original_question": "What is the capital of France?",
            "generated_queries": ["What is the capital of France?"],
            "retrieved_documents": [document],
            "documents": [document],
            "web_search_used": False,
            "query_rewritten": False,
            "citations": [source],
            "sources": [source],
            "timings": {"retrieve": 1.5, "generate": 2.5},
            "usage": {"total_tokens": 18, "estimated_cost_usd": 0.0},
        },
    )

    response = APIClient().post(
        "/api/query/",
        {
            "question": "What is the capital of France?",
            "document_id": record.document_id,
            "user_id": "user-1",
            "profile": "fast",
        },
        format="json",
    )

    assert response.status_code == 200, response.content
    payload = response.json()
    assert payload["success"] is True
    assert payload["citations"][0]["citation_id"] == "S1"
    assert payload["timings_ms"]["retrieve"] == 1.5
    assert payload["usage"]["total_tokens"] == 18
    assert payload["profile"] == "fast"
    assert QueryLog.objects.count() == 1


@pytest.mark.django_db
def test_query_missing_document_returns_404():
    response = APIClient().post(
        "/api/query/",
        {
            "question": "What is this?",
            "document_id": uuid.uuid4().hex,
            "user_id": "user-1",
        },
        format="json",
    )
    assert response.status_code == 404


@pytest.mark.django_db
def test_query_unavailable_index_returns_410(monkeypatch):
    record = DocumentRecord.objects.create(
        user_id="user-1",
        document_id=uuid.uuid4().hex,
        original_name="sample.pdf",
        stored_path="/tmp/sample.pdf",
        collection_name="rag_user-1_doc_v",
        persist_path="/tmp/missing",
        chunk_count=1,
        content_hash="b" * 64,
        embedding_model="nomic-embed-text:latest",
    )

    def fail(*args, **kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(views, "_get_retriever", fail)
    response = APIClient().post(
        "/api/query/",
        {
            "question": "What is this?",
            "document_id": record.document_id,
            "user_id": "user-1",
        },
        format="json",
    )
    assert response.status_code == 410


@pytest.mark.django_db
def test_health_check():
    response = APIClient().get("/api/health/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
