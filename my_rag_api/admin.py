from django.contrib import admin

from .models import DocumentRecord, QueryLog


@admin.register(DocumentRecord)
class DocumentRecordAdmin(admin.ModelAdmin):
    list_display = (
        "document_id",
        "original_name",
        "user_id",
        "chunk_count",
        "embedding_model",
        "created_at",
    )
    search_fields = ("document_id", "original_name", "user_id", "content_hash")
    list_filter = ("embedding_model", "created_at")


@admin.register(QueryLog)
class QueryLogAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user_id",
        "profile",
        "web_search_used",
        "retrieved_count",
        "relevant_count",
        "created_at",
    )
    search_fields = ("question", "answer", "user_id")
    list_filter = ("profile", "web_search_used", "created_at")
    readonly_fields = ("timings", "usage", "sources", "citations")
