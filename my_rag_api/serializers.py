import zipfile
from pathlib import Path

from django.conf import settings
from django.core.validators import FileExtensionValidator
from rest_framework import serializers

ALLOWED_EXTENSIONS = {"pdf", "docx", "pptx"}


class DocumentUploadSerializer(serializers.Serializer):
    """Serializer for document upload endpoint."""

    file = serializers.FileField(
        validators=[FileExtensionValidator(allowed_extensions=["pdf", "docx", "pptx"])],
        help_text="The file to upload (PDF, DOCX, or PPTX).",
    )

    user_id = serializers.RegexField(
        r"^[A-Za-z0-9_-][A-Za-z0-9._-]{0,99}$",
        default="1",
        help_text="ID of the user uploading the document.",
        error_messages={
            "invalid": "user_id may contain letters, numbers, dots, underscores, and hyphens.",
        },
    )

    def validate_user_id(self, value):
        if value in {".", ".."}:
            raise serializers.ValidationError("Invalid user_id.")
        return value

    def validate_file(self, value):
        raw_name = str(value.name or "")
        if not raw_name or "/" in raw_name or "\\" in raw_name:
            raise serializers.ValidationError("Invalid file name.")
        name = Path(raw_name).name
        if name != raw_name:
            raise serializers.ValidationError("Invalid file name.")

        extension = Path(name).suffix.lower().lstrip(".")
        if extension not in ALLOWED_EXTENSIONS:
            raise serializers.ValidationError(
                f"Unsupported file type. Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            )

        max_size = getattr(settings, "MAX_UPLOAD_SIZE_MB", 50) * 1024 * 1024
        if value.size > max_size:
            raise serializers.ValidationError(
                f"File too large. Size should not exceed {getattr(settings, 'MAX_UPLOAD_SIZE_MB', 50)} MB."
            )

        if extension == "pdf":
            self._validate_pdf(value)
        else:
            self._validate_office_zip(value, extension)
        return value

    @staticmethod
    def _validate_pdf(value):
        value.seek(0)
        header = value.read(5)
        value.seek(0)
        if header != b"%PDF-":
            raise serializers.ValidationError("File is not a valid PDF.")

    @staticmethod
    def _validate_office_zip(value, extension):
        expected_prefix = "word/" if extension == "docx" else "ppt/"
        value.seek(0)
        try:
            with zipfile.ZipFile(value) as archive:
                names = archive.namelist()
        except zipfile.BadZipFile as error:
            raise serializers.ValidationError(
                f"File is not a valid {extension.upper()} document."
            ) from error
        finally:
            value.seek(0)

        if not any(name.startswith(expected_prefix) for name in names):
            raise serializers.ValidationError(
                f"File does not contain the expected {extension.upper()} structure."
            )


class QuerySerializer(serializers.Serializer):
    """Serializer for query input."""

    question = serializers.CharField(
        max_length=500,
        trim_whitespace=True,
        help_text="The user's question to be processed.",
        error_messages={
            "required": "Question is required.",
            "blank": "Question cannot be blank.",
        },
    )

    document_id = serializers.RegexField(
        r"^[0-9a-f]{32}$",
        help_text="The UUID returned by the upload endpoint.",
        error_messages={"invalid": "document_id must be a 32-character lowercase hex UUID."},
    )

    user_id = serializers.RegexField(
        r"^[A-Za-z0-9_-][A-Za-z0-9._-]{0,99}$",
        default="1",
        help_text="ID of the user making the query.",
    )

    profile = serializers.ChoiceField(
        choices=["fast", "balanced", "quality"],
        required=False,
        allow_null=True,
        default=None,
        help_text="Runtime profile controlling retrieval, grading, and web fallback.",
    )

    enable_web_search = serializers.BooleanField(
        required=False,
        allow_null=True,
        default=None,
        help_text="Whether to enable web search fallback. Defaults to the selected profile.",
    )

    def validate_question(self, value):
        if not value.strip():
            raise serializers.ValidationError("Question cannot be empty or whitespace.")
        return value.strip()

    def validate_user_id(self, value):
        if value in {".", ".."}:
            raise serializers.ValidationError("Invalid user_id.")
        return value


class DocumentResponseSerializer(serializers.Serializer):
    """Serializer for document upload responses."""

    success = serializers.BooleanField()
    message = serializers.CharField()
    document_id = serializers.CharField()
    original_name = serializers.CharField()
    chunks_created = serializers.IntegerField(required=False)
    content_hash = serializers.CharField(required=False)
    user_id = serializers.CharField(required=False)


class QueryResponseSerializer(serializers.Serializer):
    """Serializer for query responses."""

    success = serializers.BooleanField()
    document_id = serializers.CharField()
    profile = serializers.CharField()
    original_query = serializers.CharField()
    answer = serializers.CharField()
    generated_queries = serializers.ListField(child=serializers.CharField(), required=False)
    retrieved_documents = serializers.IntegerField(required=False)
    relevant_documents = serializers.IntegerField(required=False)
    web_search_used = serializers.BooleanField(required=False)
    query_rewritten = serializers.BooleanField(required=False)
    citations = serializers.ListField(child=serializers.DictField(), required=False)
    sources = serializers.ListField(child=serializers.DictField(), required=False)
    timings_ms = serializers.DictField(required=False)
    usage = serializers.DictField(required=False)


class ErrorResponseSerializer(serializers.Serializer):
    """Serializer for error responses."""

    success = serializers.BooleanField(default=False)
    error = serializers.CharField()
    details = serializers.CharField(required=False)
