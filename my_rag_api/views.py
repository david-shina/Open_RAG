import hashlib
import logging
import threading
import uuid
from pathlib import Path

from django.conf import settings
from django.db import connection
from django.views.decorators.csrf import csrf_exempt
from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response

from rag_demo.embeddings import Embedding

from .models import DocumentRecord, QueryLog
from .serializers import (
    DocumentResponseSerializer,
    DocumentUploadSerializer,
    ErrorResponseSerializer,
    QueryResponseSerializer,
    QuerySerializer,
)
from .utilis import RAGConfig, loadingDocument, run_rag_agent

logger = logging.getLogger(__name__)

_vector_store_cache: dict[tuple[str, int], object] = {}
_vector_store_lock = threading.Lock()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get_retriever(record: DocumentRecord, top_k: int):
    cache_key = (record.document_id, top_k)
    with _vector_store_lock:
        cached = _vector_store_cache.get(cache_key)
    if cached is not None:
        return cached

    retriever = Embedding.load_retriever(
        persist_path=record.persist_path,
        collection_name=record.collection_name,
        model_name=record.embedding_model or settings.EMBEDDING_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        knn=top_k,
    )
    with _vector_store_lock:
        _vector_store_cache[cache_key] = retriever
    return retriever


@extend_schema(
    summary="Upload a document",
    description="Validate, store, chunk, embed, and index a PDF, DOCX, or PPTX document.",
    request={
        "multipart/form-data": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "format": "binary"},
                "user_id": {"type": "string"},
            },
            "required": ["file", "user_id"],
        }
    },
    responses={201: DocumentResponseSerializer, 400: ErrorResponseSerializer, 500: ErrorResponseSerializer},
)
@csrf_exempt
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def upload_document(request):
    file_path: Path | None = None
    record: DocumentRecord | None = None
    embedder: Embedding | None = None
    document_id: str | None = None
    user_id: str | None = None

    try:
        serializer = DocumentUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {"success": False, "error": "Invalid upload", "details": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST,
            )

        uploaded_file = serializer.validated_data["file"]
        user_id = serializer.validated_data["user_id"]
        original_name = Path(str(uploaded_file.name)).name
        extension = Path(original_name).suffix.lower()
        document_id = uuid.uuid4().hex

        user_dir = Path(settings.RAG_UPLOAD_DIR) / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        file_path = user_dir / f"{document_id}{extension}"

        with open(file_path, "wb+") as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        content_hash = _sha256_file(file_path)
        logger.info("Saved upload %s for user %s at %s", document_id, user_id, file_path)

        chunks = loadingDocument(str(file_path))
        for chunk in chunks:
            chunk.metadata = dict(chunk.metadata or {})
            chunk.metadata["source"] = original_name
            chunk.metadata["sourceId"] = original_name
        logger.info("Chunked document %s into %s chunks", document_id, len(chunks))

        embedder = Embedding(
            path=str(file_path),
            chunks=chunks,
            persist_root=settings.RAG_CHROMA_DIR,
            model_name=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )
        embedder.set_embedding_model(
            model_type=settings.EMBEDDING_PROVIDER,
            model_name=settings.EMBEDDING_MODEL,
        )
        result = embedder.create_or_load_vectorstore(
            userId=str(user_id),
            documentId=document_id,
            knn=int(getattr(settings, "RAG_TOP_K", 5)),
        )

        record = DocumentRecord.objects.create(
            user_id=str(user_id),
            document_id=document_id,
            original_name=original_name,
            stored_path=str(file_path),
            collection_name=result["collection_name"],
            persist_path=result["persist_path"],
            chunk_count=result.get("chunk_count", len(chunks)),
            content_hash=content_hash,
            embedding_model=settings.EMBEDDING_MODEL,
        )

        cache_key = (document_id, int(getattr(settings, "RAG_TOP_K", 5)))
        with _vector_store_lock:
            _vector_store_cache[cache_key] = result["retriever"]

        response_data = {
            "success": True,
            "message": "Document uploaded and processed successfully",
            "document_id": document_id,
            "original_name": original_name,
            "chunks_created": len(chunks),
            "content_hash": content_hash,
            "user_id": user_id,
        }
        return Response(response_data, status=status.HTTP_201_CREATED)

    except ValueError as error:
        logger.warning("Document processing failed: %s", error)
        if record is not None:
            record.delete()
        if embedder is not None and user_id and document_id:
            embedder.cleanup_vectorstore(user_id, document_id)
        if file_path is not None:
            file_path.unlink(missing_ok=True)
        return Response(
            {"success": False, "error": "Failed to process document", "details": str(error)},
            status=status.HTTP_400_BAD_REQUEST,
        )
    except Exception as error:
        logger.error("Unexpected error in upload_document: %s", error, exc_info=True)
        if record is not None:
            record.delete()
        if embedder is not None and user_id and document_id:
            embedder.cleanup_vectorstore(user_id, document_id)
        if file_path is not None:
            file_path.unlink(missing_ok=True)
        return Response(
            {"success": False, "error": "An unexpected error occurred", "details": str(error)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@extend_schema(
    summary="Query a document",
    description=(
        "Run the adaptive RAG pipeline with query translation, document grading, "
        "optional web fallback, citations, stage timings, and token usage."
    ),
    request=QuerySerializer,
    responses={
        200: QueryResponseSerializer,
        404: ErrorResponseSerializer,
        410: ErrorResponseSerializer,
        500: ErrorResponseSerializer,
    },
)
@api_view(["POST"])
def query_document(request):
    try:
        serializer = QuerySerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {"success": False, "error": "Invalid query", "details": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST,
            )

        user_id = serializer.validated_data["user_id"]
        document_id = serializer.validated_data["document_id"]
        question = serializer.validated_data["question"]
        profile = serializer.validated_data.get("profile") or getattr(
            settings, "RAG_PROFILE", "balanced"
        )
        enable_web_search = serializer.validated_data.get("enable_web_search")

        record = DocumentRecord.objects.filter(
            user_id=str(user_id),
            document_id=document_id,
        ).first()
        if record is None:
            return Response(
                {
                    "success": False,
                    "error": "Document not found or not processed for this user.",
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        rag_config = RAGConfig.from_env(profile=profile)
        try:
            retriever = _get_retriever(record, rag_config.top_k)
        except Exception as error:
            logger.error("Failed to load index for %s: %s", document_id, error, exc_info=True)
            return Response(
                {
                    "success": False,
                    "error": "Document index is unavailable. Please re-upload the document.",
                },
                status=status.HTTP_410_GONE,
            )

        final_state = run_rag_agent(
            retriever,
            question,
            enable_web_search=enable_web_search,
            rag_config=rag_config,
        )

        answer = final_state.get(
            "generation",
            "I couldn't find any relevant information in the document to answer your question.",
        )
        timings = final_state.get("timings", {})
        usage = final_state.get("usage", {})
        sources = final_state.get("sources", [])
        citations = final_state.get("citations", [])
        retrieved_count = len(final_state.get("retrieved_documents", []))
        relevant_count = len(final_state.get("documents", []))

        try:
            QueryLog.objects.create(
                document=record,
                user_id=str(user_id),
                question=question,
                answer=answer,
                profile=rag_config.profile,
                web_search_used=bool(final_state.get("web_search_used", False)),
                retrieved_count=retrieved_count,
                relevant_count=relevant_count,
                timings=timings,
                usage=usage,
                sources=sources,
                citations=citations,
            )
        except Exception as error:
            logger.warning("Failed to persist query log: %s", error)

        return Response(
            {
                "success": True,
                "document_id": document_id,
                "profile": rag_config.profile,
                "original_query": final_state.get("original_question", question),
                "answer": answer,
                "generated_queries": final_state.get("generated_queries", []),
                "retrieved_documents": retrieved_count,
                "relevant_documents": relevant_count,
                "web_search_used": bool(final_state.get("web_search_used", False)),
                "query_rewritten": bool(final_state.get("query_rewritten", False)),
                "citations": citations,
                "sources": sources,
                "timings_ms": timings,
                "usage": usage,
            },
            status=status.HTTP_200_OK,
        )

    except Exception as error:
        logger.error("Error in query_document: %s", error, exc_info=True)
        return Response(
            {"success": False, "error": str(error)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@extend_schema(
    summary="Health check",
    description="Check whether the API and database are reachable.",
    responses={200: dict, 503: dict},
)
@api_view(["GET"])
def health_check(request):
    try:
        connection.ensure_connection()
        database_ok = True
    except Exception:
        database_ok = False
    status_code = status.HTTP_200_OK if database_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return Response(
        {
            "status": "ok" if database_ok else "degraded",
            "service": "open-rag",
            "database": database_ok,
        },
        status=status_code,
    )
