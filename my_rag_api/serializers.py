from pathlib import Path
from rest_framework import serializers
from django.core.validators import FileExtensionValidator

class DocumentUploadSerializer(serializers.Serializer):

    """Serializer for document upload endpoint."""
    file = serializers.FileField(
        validators=[FileExtensionValidator(allowed_extensions=['pdf', 'docx', 'pptx'])], 
        help_text="The file to upload (PDF, DOCX, or PPTX)."
    )

    user_id = serializers.CharField(max_length=100, default=1, help_text="ID of the user uploading the document.")

    def validate_file(self, value):
        """Custom validation for the uploaded file."""
   
        
        if value.size > 50 * 1024 * 1024:
            raise serializers.ValidationError(
                f"File too large. Size should not exceed {50} MB."
            )

        # 2. Check file extension using pathlib
        # ext = Path(value.name).suffix.lower()
        # valid_extensions = {'.pdf', '.docx', '.pptx'}

        # if ext not in valid_extensions:
        #     raise serializers.ValidationError(
        #         f"Unsupported file type. Allowed types: {', '.join(valid_extensions)}"
        #     )

        return value




class QuerySerializer(serializers.Serializer):
    """Serializer for query input."""
    question = serializers.CharField(
        max_length=500, 
        help_text="The user's question to be processed.",
        error_messages={
            'required': "Question is required.",
            'blank': "Question cannot be blank.",
        }
    )


    document_id  = serializers.CharField(
        max_length=255,
        help_text="The ID of the document to query against.",
    )


    user_id = serializers.CharField(
        max_length=100,
        default=1,
        help_text="ID of the user making the query."
    )


    enable_web_search = serializers.BooleanField(
        default=True,
        help_text="Whether to enable web search for the query."
    )


    def validate_question(self, value):

        """Custom validation for the question field."""
        if not value.strip():
            raise serializers.ValidationError("Question cannot be empty or whitespace.")
        return value.strip()
    


class DocumentResponseSerializer(serializers.Serializer):
    """Serializer for document response"""


    success = serializers.BooleanField(help_text="Indicates if the operation was successful.")
    message = serializers.CharField()
    document_id = serializers.CharField(required=False, help_text="ID of the processed document.")
    document_path = serializers.CharField(required=False, help_text="Path to the processed document.")
    chunks_created = serializers.IntegerField(required=False, help_text="Number of chunks created from the document.")
    user_id = serializers.CharField(required=False, help_text="ID of the user associated with the document.")



class QueryResponseSerializer(serializers.Serializer):
    """
    Serializer for query response
    """
    success = serializers.BooleanField()
    question = serializers.CharField()
    answer = serializers.CharField()
    documents_retrieved = serializers.IntegerField(required=False)
    web_search_used = serializers.BooleanField(required=False)
    sources = serializers.ListField(
        child=serializers.DictField(),
        required=False,
        help_text='List of source documents used'
    )





class ErrorResponseSerializer(serializers.Serializer):
    """Serializer for error responses."""
    success = serializers.BooleanField(default=False, help_text="Indicates if the operation was successful.")
    error = serializers.CharField(help_text="Description of the error that occurred.")
    details = serializers.CharField(required=False, help_text="Additional details about the error.")