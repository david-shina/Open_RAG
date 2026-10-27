from my_rag_api import utilis

from .fakes import FakeChatModel, FakeGrader, FakeRetriever, FakeTranslator


def test_deduplicate_documents(sample_documents):
    duplicated = sample_documents + [sample_documents[0]]
    unique = utilis._deduplicate_documents(duplicated)
    assert len(unique) == 2


def test_extract_sources_deduplicates_by_chunk(sample_documents):
    documents = utilis._assign_citation_ids(sample_documents)
    sources = utilis._extract_sources(documents + [documents[0]])
    assert [source["citation_id"] for source in sources] == ["S1", "S2"]
    assert sources[0]["source"] == "paris.pdf"


def test_extract_citations_supports_grouped_labels(sample_documents):
    documents = utilis._assign_citation_ids(sample_documents)
    citations = utilis._extract_citations("First [S1], second [S1, S2].", documents)
    assert [citation["citation_id"] for citation in citations] == ["S1", "S2"]


def test_retrieve_uses_translation_and_deduplicates(monkeypatch, sample_documents):
    retriever = FakeRetriever(sample_documents)
    monkeypatch.setattr(
        utilis,
        "_query_translator",
        lambda *args, **kwargs: FakeTranslator(["variant one", "variant two"]),
    )
    rag_config = utilis.RAGConfig(
        profile="balanced",
        translation_mode="multiquery",
        parallel_retrieval=True,
        max_workers=2,
        request_timeout=1,
    )
    state = {"question": "When was the Eiffel Tower built?"}
    result = utilis.retrieve(
        state,
        {"configurable": {"retriever": retriever, "rag_config": rag_config}},
    )

    assert set(retriever.queries) == {
        "When was the Eiffel Tower built?",
        "variant one",
        "variant two",
    }
    assert result["generated_queries"] == [
        "When was the Eiffel Tower built?",
        "variant one",
        "variant two",
    ]
    assert len(result["retrieved_documents"]) == 2
    assert result["documents"][0].metadata["citation_id"] == "S1"
    assert "retrieve" in result["timings"]


def test_grade_documents_filters_and_requests_web(monkeypatch, sample_documents):
    grader = FakeGrader(["no", "no"])
    monkeypatch.setattr(utilis, "_grader_chain", lambda *args, **kwargs: grader)
    rag_config = utilis.RAGConfig(grading_mode="llm", max_grade_docs=2)
    result = utilis.grade_documents(
        {"question": "What is the capital of France?", "documents": sample_documents, "web_search_enabled": True},
        {"configurable": {"rag_config": rag_config}},
    )

    assert result["documents"] == []
    assert result["web_search_used"] is True
    assert result["sources"] == []
    assert len(grader.calls) == 2


def test_grade_documents_keeps_relevant_docs(monkeypatch, sample_documents):
    grader = FakeGrader(["yes", "no"])
    monkeypatch.setattr(utilis, "_grader_chain", lambda *args, **kwargs: grader)
    rag_config = utilis.RAGConfig(grading_mode="llm", max_grade_docs=2)
    result = utilis.grade_documents(
        {"question": "Where is the Eiffel Tower?", "documents": sample_documents, "web_search_enabled": True},
        {"configurable": {"rag_config": rag_config}},
    )

    assert len(result["documents"]) == 1
    assert result["web_search_used"] is False
    assert result["sources"][0]["source"] == "paris.pdf"


def test_grade_documents_without_docs_requests_web():
    result = utilis.grade_documents(
        {"question": "What happened in 2024?", "documents": [], "web_search_enabled": True},
        {"configurable": {"rag_config": utilis.RAGConfig()}},
    )

    assert result["documents"] == []
    assert result["web_search_used"] is True


def test_generate_extracts_citations_and_usage(monkeypatch, sample_documents):
    documents = utilis._assign_citation_ids(sample_documents)
    fake_model = FakeChatModel("The Eiffel Tower was completed in 1889 [S1].")
    monkeypatch.setattr(utilis, "_llm", lambda config: fake_model)
    rag_config = utilis.RAGConfig()
    result = utilis.generate(
        {"question": "When was the Eiffel Tower built?", "documents": documents},
        {"configurable": {"rag_config": rag_config}},
    )

    assert result["generation"] == "The Eiffel Tower was completed in 1889 [S1]."
    assert [citation["citation_id"] for citation in result["citations"]] == ["S1"]
    assert result["citations"][0]["source"] == "paris.pdf"
    assert result["usage"]["total_tokens"] == 18
    assert result["usage"]["estimated_cost_usd"] == 0.0
    assert "generate" in result["timings"]
    assert fake_model.calls


def test_web_search_without_key_preserves_local_documents(monkeypatch, sample_documents):
    monkeypatch.setattr(utilis, "_web_search_tool", lambda: None)
    result = utilis.web_search(
        {"question": "What is new?", "documents": sample_documents},
        {"configurable": {"rag_config": utilis.RAGConfig()}},
    )

    assert result["web_search_used"] is False
    assert len(result["documents"]) == 2


def test_run_rag_agent_fast_profile(monkeypatch, sample_documents):
    monkeypatch.setattr(utilis, "_llm", lambda config: FakeChatModel("Completed in 1889 [S1]."))
    rag_config = utilis.RAGConfig(
        profile="fast",
        translation_mode="none",
        grading_mode="none",
        enable_web_search=False,
        parallel_retrieval=False,
    )
    retriever = FakeRetriever(sample_documents)

    result = utilis.run_rag_agent(
        retriever,
        "When was the Eiffel Tower built?",
        enable_web_search=False,
        rag_config=rag_config,
    )

    assert result["generation"] == "Completed in 1889 [S1]."
    assert result["citations"][0]["source"] == "paris.pdf"
    assert result["usage"]["total_tokens"] == 18
    assert set(result["timings"]) >= {"retrieve", "grade_documents", "generate"}
    assert result["web_search_used"] is False

def test_extract_citations_supports_fullwidth_labels(sample_documents):
    documents = utilis._assign_citation_ids(sample_documents)
    citations = utilis._extract_citations("The tower is in Paris 【S1】.", documents)
    assert [citation["citation_id"] for citation in citations] == ["S1"]


def test_normalize_citations_converts_fullwidth_brackets():
    assert utilis._normalize_citations("Paris 【S1】 and 【S2】.") == "Paris [S1] and [S2]."
