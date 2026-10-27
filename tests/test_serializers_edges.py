import io
import zipfile

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework import serializers

from my_rag_api.serializers import DocumentUploadSerializer, QuerySerializer


def test_document_upload_rejects_invalid_docx_zip():
    upload = SimpleUploadedFile("report.docx", b"not a zip", content_type="application/octet-stream")
    serializer = DocumentUploadSerializer(data={"file": upload, "user_id": "user-1"})
    assert not serializer.is_valid()


def test_document_upload_rejects_zip_without_word_structure():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types />")
    upload = SimpleUploadedFile("report.docx", buffer.getvalue())
    serializer = DocumentUploadSerializer(data={"file": upload, "user_id": "user-1"})
    assert not serializer.is_valid()


def test_document_upload_rejects_oversized_file(monkeypatch):
    from my_rag_api import serializers as serializer_module

    monkeypatch.setattr(serializer_module.settings, "MAX_UPLOAD_SIZE_MB", 0)
    upload = SimpleUploadedFile("report.pdf", b"%PDF-1.4\nbody", content_type="application/pdf")
    serializer = DocumentUploadSerializer(data={"file": upload, "user_id": "user-1"})
    assert not serializer.is_valid()


def test_query_serializer_rejects_blank_question():
    serializer = QuerySerializer(
        data={"question": "   ", "document_id": "a" * 32, "user_id": "user-1"}
    )
    assert not serializer.is_valid()
    assert "question" in serializer.errors


def test_query_serializer_rejects_path_traversal_user_id():
    serializer = QuerySerializer(
        data={"question": "Question?", "document_id": "a" * 32, "user_id": ".."}
    )
    assert not serializer.is_valid()
    assert "user_id" in serializer.errors


def test_validate_file_rejects_empty_name():
    serializer = DocumentUploadSerializer()

    class EmptyName:
        name = ""
        size = 1

        def seek(self, position):
            return 0

        def read(self, size=-1):
            return b""

    with pytest.raises(serializers.ValidationError):
        serializer.validate_file(EmptyName())
