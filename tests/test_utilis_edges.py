import pytest

from my_rag_api import utilis

from .fakes import FakeChatModel, FakeRetriever, FakeTranslator


class RaisingChain:
    def invoke(self, *args, **kwargs):
        raise RuntimeError("boom")


class FakeWebTool:
    def __init__(self):
        self.queries = []

    def search(self, query, max_results=5):
        self.queries.append(query)
        return {
            "results": [
                {
                    "url": "https://example.com/web",
                    "content": "The web result contains a current answer.",
                }
            ]
        }


class FakeRewriter:
    def invoke(self, payload):
        return "rewritten question"


class FakeResponse:
    usage_metadata = None
    response_metadata = {
        "token_usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
    }


def test_retrieve_without_retriever():
    result = utilis.retrieve(
        {"question": "What is this?"},
        {"configurable": {"rag_config": utilis.RAGConfig()}},
    )
    assert result["documents"] == []
    assert result["generated_queries"] == ["What is this?"]
    assert "retrieve" in result["timings"]


def test_retrieve_falls_back_when_translation_fails(monkeypatch, sample_documents):
    monkeypatch.setattr(utilis, "_query_translator", lambda *args, **kwargs: RaisingChain())
    retriever = FakeRetriever(sample_documents)
    result = utilis.retrieve(
        {"question": "Original question"},
        {
            "configurable": {
                "retriever": retriever,
                "rag_config": utilis.RAGConfig(translation_mode="multiquery", parallel_retrieval=False),
            }
        },
    )
    assert result["generated_queries"] == ["Original question"]
    assert len(result["documents"]) == 2


def test_grading_failure_keeps_documents(monkeypatch, sample_documents):
    monkeypatch.setattr(utilis, "_grader_chain", lambda *args, **kwargs: RaisingChain())
    result = utilis.grade_documents(
        {"question": "Question", "documents": sample_documents, "web_search_enabled": True},
        {"configurable": {"rag_config": utilis.RAGConfig(grading_mode="llm")}},
    )
    assert len(result["documents"]) == 2
    assert result["web_search_used"] is False


def test_grading_none_keeps_documents(sample_documents):
    result = utilis.grade_documents(
        {"question": "Question", "documents": sample_documents, "web_search_enabled": True},
        {"configurable": {"rag_config": utilis.RAGConfig(grading_mode="none")}},
    )
    assert len(result["documents"]) == 2


def test_broad_question_skips_grading(monkeypatch, sample_documents):
    monkeypatch.setattr(utilis, "_grader_chain", lambda *args, **kwargs: RaisingChain())
    result = utilis.grade_documents(
        {"question": "Summarize this paper", "documents": sample_documents, "web_search_enabled": True},
        {"configurable": {"rag_config": utilis.RAGConfig(grading_mode="llm")}},
    )
    assert len(result["documents"]) == 2
    assert result["web_search_used"] is False


def test_transform_query_failure_keeps_question(monkeypatch):
    monkeypatch.setattr(utilis, "_rewriter_chain", lambda *args, **kwargs: RaisingChain())
    result = utilis.transform_query(
        {"question": "original", "documents": []},
        {"configurable": {"rag_config": utilis.RAGConfig()}},
    )
    assert result["question"] == "original"
    assert result["query_rewritten"] is True


def test_web_search_success(monkeypatch, sample_documents):
    tool = FakeWebTool()
    monkeypatch.setattr(utilis, "_web_search_tool", lambda: tool)
    result = utilis.web_search(
        {"question": "current question", "documents": sample_documents},
        {"configurable": {"rag_config": utilis.RAGConfig()}},
    )
    assert result["web_search_used"] is True
    assert len(result["documents"]) == 3
    assert any(source["url"] == "https://example.com/web" for source in result["sources"])
    assert tool.queries == ["current question"]


def test_generate_failure_returns_safe_message(monkeypatch, sample_documents):
    monkeypatch.setattr(utilis, "_llm", lambda config: RaisingChain())
    result = utilis.generate(
        {"question": "Question", "documents": sample_documents},
        {"configurable": {"rag_config": utilis.RAGConfig()}},
    )
    assert "error occurred" in result["generation"]
    assert result["usage"] == {}
    assert result["citations"] == []


def test_usage_from_response_metadata():
    config = utilis.RAGConfig(input_cost_per_million=1.0, output_cost_per_million=2.0)
    usage = utilis._usage_from_response(FakeResponse(), config)
    assert usage["prompt_tokens"] == 100
    assert usage["completion_tokens"] == 50
    assert usage["total_tokens"] == 150
    assert usage["estimated_cost_usd"] == 0.0002


def test_message_text_handles_multipart_content():
    assert utilis._message_text([{"text": "hello "}, {"text": "world"}]) == "hello world"


def test_chat_reports_missing_file():
    assert utilis.chat("does-not-exist.pdf", "question") == "Error: File path does not exist."


def test_run_rag_agent_uses_web_fallback(monkeypatch):
    tool = FakeWebTool()
    monkeypatch.setattr(utilis, "_query_translator", lambda *args, **kwargs: FakeTranslator(["variant"]))
    monkeypatch.setattr(utilis, "_rewriter_chain", lambda *args, **kwargs: FakeRewriter())
    monkeypatch.setattr(utilis, "_web_search_tool", lambda: tool)
    monkeypatch.setattr(utilis, "_llm", lambda config: FakeChatModel("Web answer [S1]."))
    config = utilis.RAGConfig(
        profile="balanced",
        translation_mode="multiquery",
        grading_mode="llm",
        enable_web_search=True,
        parallel_retrieval=False,
    )

    result = utilis.run_rag_agent(
        FakeRetriever([]),
        "What is current?",
        enable_web_search=True,
        rag_config=config,
    )

    assert result["web_search_used"] is True
    assert result["query_rewritten"] is True
    assert result["generation"] == "Web answer [S1]."
    assert result["citations"][0]["url"] == "https://example.com/web"
    assert tool.queries == ["rewritten question"]


def test_loading_document_dispatches_pdf(monkeypatch, sample_documents):
    class FakePDFLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            return sample_documents

    monkeypatch.setattr(utilis.pdf_loader, "PDFLoader", FakePDFLoader)
    assert utilis.loadingDocument("document.pdf") == sample_documents


def test_loading_document_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "document.txt"
    path.write_text("text", encoding="utf-8")
    with pytest.raises(ValueError):
        utilis.loadingDocument(str(path))


def test_loading_document_rejects_empty_extraction(monkeypatch):
    class EmptyPDFLoader:
        def __init__(self, path):
            pass

        def load(self):
            return []

    monkeypatch.setattr(utilis.pdf_loader, "PDFLoader", EmptyPDFLoader)
    with pytest.raises(ValueError):
        utilis.loadingDocument("document.pdf")


def test_embedding_helper_returns_retriever(monkeypatch, sample_documents):
    class FakeEmbedding:
        def __init__(self, path, chunks):
            self.path = path
            self.chunks = chunks

        def create_or_load_vectorstore(self, **kwargs):
            return {"retriever": "retriever"}

    monkeypatch.setattr(utilis, "Embedding", FakeEmbedding)
    assert utilis.embedding("document.pdf", sample_documents) == "retriever"

def test_message_text_handles_plain_parts():
    assert utilis._message_text(["hello ", "world"]) == "hello world"
