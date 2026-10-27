from django.db import models


class DocumentRecord(models.Model):
    user_id = models.CharField(max_length=100, db_index=True)
    document_id = models.CharField(max_length=64, unique=True)
    original_name = models.CharField(max_length=255)
    stored_path = models.CharField(max_length=1024)
    collection_name = models.CharField(max_length=255)
    persist_path = models.CharField(max_length=1024)
    chunk_count = models.PositiveIntegerField(default=0)
    content_hash = models.CharField(max_length=64, db_index=True)
    embedding_model = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user_id", "document_id"]),
        ]

    def __str__(self) -> str:
        return f"{self.original_name} ({self.document_id})"


class QueryLog(models.Model):
    document = models.ForeignKey(
        DocumentRecord,
        on_delete=models.CASCADE,
        related_name="queries",
    )
    user_id = models.CharField(max_length=100, db_index=True)
    question = models.TextField()
    answer = models.TextField(blank=True)
    profile = models.CharField(max_length=32, default="balanced")
    web_search_used = models.BooleanField(default=False)
    retrieved_count = models.PositiveIntegerField(default=0)
    relevant_count = models.PositiveIntegerField(default=0)
    timings = models.JSONField(default=dict, blank=True)
    usage = models.JSONField(default=dict, blank=True)
    sources = models.JSONField(default=list, blank=True)
    citations = models.JSONField(default=list, blank=True)
    feedback = models.SmallIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user_id", "created_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.user_id}: {self.question[:60]}"
