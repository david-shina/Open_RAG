import io
import uuid
import zipfile

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework import serializers

from my_rag_api.serializers import DocumentUploadSerializer, QuerySerializer


def _office_file(name, prefix):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types />")
        archive.writestr(f"{prefix}document.xml", "<document />")
    return SimpleUploadedFile(name, buffer.getvalue())


def test_document_upload_accepts_pdf():
    upload = SimpleUploadedFile("report.pdf", b"%PDF-1.4\nbody\n%%EOF", content_type="application/pdf")
    serializer = DocumentUploadSerializer(data={"file": upload, "user_id": "user-1"})
    assert serializer.is_valid(), serializer.errors


def test_document_upload_rejects_fake_pdf():
    upload = SimpleUploadedFile("report.pdf", b"not a pdf", content_type="application/pdf")
    serializer = DocumentUploadSerializer(data={"file": upload, "user_id": "user-1"})
    assert not serializer.is_valid()
    assert "file" in serializer.errors


def test_document_upload_accepts_docx_and_pptx():
    docx = _office_file("report.docx", "word/")
    pptx = _office_file("slides.pptx", "ppt/")
    assert DocumentUploadSerializer(data={"file": docx, "user_id": "user-1"}).is_valid()
    assert DocumentUploadSerializer(data={"file": pptx, "user_id": "user-1"}).is_valid()


class FakeUpload:
    def __init__(self, name, data):
        self.name = name
        self.size = len(data)
        self._stream = io.BytesIO(data)

    def seek(self, position):
        return self._stream.seek(position)

    def read(self, size=-1):
        return self._stream.read(size)


def test_document_upload_rejects_path_traversal_filename():
    upload = FakeUpload("../report.pdf", b"%PDF-1.4\nbody")
    serializer = DocumentUploadSerializer()
    with pytest.raises(serializers.ValidationError):
        serializer.validate_file(upload)


def test_document_upload_rejects_invalid_user_id():
    upload = SimpleUploadedFile("report.pdf", b"%PDF-1.4\nbody", content_type="application/pdf")
    serializer = DocumentUploadSerializer(data={"file": upload, "user_id": ".."})
    assert not serializer.is_valid()
    assert "user_id" in serializer.errors


def test_query_serializer_accepts_valid_payload():
    serializer = QuerySerializer(
        data={
            "question": "What is this?",
            "document_id": uuid.uuid4().hex,
            "user_id": "user-1",
            "profile": "fast",
            "enable_web_search": False,
        }
    )
    assert serializer.is_valid(), serializer.errors


def test_query_serializer_rejects_invalid_document_id_and_profile():
    serializer = QuerySerializer(
        data={
            "question": "What is this?",
            "document_id": "not-a-uuid",
            "user_id": "user-1",
            "profile": "turbo",
        }
    )
    assert not serializer.is_valid()
    assert "document_id" in serializer.errors
    assert "profile" in serializer.errors
