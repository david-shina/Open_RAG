import uuid

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APIClient

from my_rag_api import views
from my_rag_api.models import DocumentRecord, QueryLog


def _record():
    return DocumentRecord.objects.create(
        user_id="user-1",
        document_id=uuid.uuid4().hex,
        original_name="sample.pdf",
        stored_path="/tmp/sample.pdf",
        collection_name="rag_user-1_doc_v",
        persist_path="/tmp/chroma",
        chunk_count=1,
        content_hash="c" * 64,
        embedding_model="nomic-embed-text:latest",
    )


@pytest.mark.django_db
def test_upload_processing_error_returns_400(tmp_path, monkeypatch):
    def fail(path):
        raise ValueError("No text could be extracted")

    monkeypatch.setattr(views, "loadingDocument", fail)
    monkeypatch.setattr(views.settings, "RAG_UPLOAD_DIR", tmp_path / "uploads")
    upload = SimpleUploadedFile("sample.pdf", b"%PDF-1.4\nbody", content_type="application/pdf")

    response = APIClient().post("/api/upload/", {"file": upload, "user_id": "user-1"}, format="multipart")

    assert response.status_code == 400
    assert response.json()["success"] is False
    assert DocumentRecord.objects.count() == 0


@pytest.mark.django_db
def test_upload_unexpected_error_returns_500(tmp_path, monkeypatch):
    def fail(path):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(views, "loadingDocument", fail)
    monkeypatch.setattr(views.settings, "RAG_UPLOAD_DIR", tmp_path / "uploads")
    upload = SimpleUploadedFile("sample.pdf", b"%PDF-1.4\nbody", content_type="application/pdf")

    response = APIClient().post("/api/upload/", {"file": upload, "user_id": "user-1"}, format="multipart")

    assert response.status_code == 500


@pytest.mark.django_db
def test_query_validation_error_returns_400():
    response = APIClient().post("/api/query/", {"question": ""}, format="json")
    assert response.status_code == 400


@pytest.mark.django_db
def test_query_unexpected_error_returns_500(monkeypatch):
    record = _record()
    monkeypatch.setattr(views, "_get_retriever", lambda record, top_k: object())
    monkeypatch.setattr(views, "run_rag_agent", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    response = APIClient().post(
        "/api/query/",
        {
            "question": "What is this?",
            "document_id": record.document_id,
            "user_id": "user-1",
        },
        format="json",
    )
    assert response.status_code == 500


@pytest.mark.django_db
def test_query_log_failure_does_not_break_response(monkeypatch):
    record = _record()
    monkeypatch.setattr(views, "_get_retriever", lambda record, top_k: object())
    monkeypatch.setattr(
        views,
        "run_rag_agent",
        lambda *args, **kwargs: {
            "generation": "Answer [S1].",
            "original_question": "Question?",
            "generated_queries": ["Question?"],
            "retrieved_documents": [],
            "documents": [],
            "web_search_used": False,
            "query_rewritten": False,
            "citations": [],
            "sources": [],
            "timings": {},
            "usage": {},
        },
    )
    monkeypatch.setattr(QueryLog.objects, "create", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("db")))

    response = APIClient().post(
        "/api/query/",
        {
            "question": "Question?",
            "document_id": record.document_id,
            "user_id": "user-1",
        },
        format="json",
    )
    assert response.status_code == 200


@pytest.mark.django_db
def test_model_string_representations():
    record = _record()
    assert str(record) == f"sample.pdf ({record.document_id})"
    log = QueryLog.objects.create(document=record, user_id="user-1", question="Question?")
    assert str(log).startswith("user-1: Question?")

def test_get_retriever_caches_instances(monkeypatch):
    from types import SimpleNamespace

    calls = []

    def fake_load(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(views.Embedding, "load_retriever", fake_load)
    views._vector_store_cache.clear()
    record = SimpleNamespace(
        document_id="doc-1",
        persist_path="/tmp/chroma",
        collection_name="rag_doc_v",
        embedding_model="nomic-embed-text:latest",
    )

    first = views._get_retriever(record, 5)
    second = views._get_retriever(record, 5)

    assert first is second
    assert len(calls) == 1
    views._vector_store_cache.clear()
