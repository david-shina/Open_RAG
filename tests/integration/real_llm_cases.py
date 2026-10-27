import os

os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from my_rag_api import utilis
from my_rag_api.utilis import RAGConfig, run_rag_agent
from rag_demo.queryTranslator import QueryTranslator

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TRANSLATOR_MODEL = os.getenv("GROQ_TRANSLATOR_MODEL", "openai/gpt-oss-20b")
REQUEST_TIMEOUT = float(os.getenv("RAG_REQUEST_TIMEOUT", "60"))
QUESTION = "What is the capital of France?"


def relevant_document() -> Document:
    return Document(
        page_content="Paris is the capital of France and its largest city.",
        metadata={"source": "france.md", "page": 1, "chunk_index": 0},
    )


def irrelevant_document() -> Document:
    return Document(
        page_content="Bananas are yellow and rich in potassium.",
        metadata={"source": "fruit.md", "page": 1, "chunk_index": 0},
    )


class StaticRetriever:
    def __init__(self, documents):
        self.documents = list(documents)

    def invoke(self, query):
        return list(self.documents)


@dataclass
class CaseResult:
    name: str
    passed: bool
    latency_ms: float
    usage: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def base_config(**overrides) -> RAGConfig:
    values = {
        "llm_model": MODEL,
        "translator_model": TRANSLATOR_MODEL,
        "temperature": 0.0,
        "request_timeout": REQUEST_TIMEOUT,
        "parallel_retrieval": False,
    }
    values.update(overrides)
    return RAGConfig(**values)


def case_translation() -> dict[str, Any]:
    translator = QueryTranslator(
        llm=ChatGroq(model=TRANSLATOR_MODEL, temperature=0.7, timeout=REQUEST_TIMEOUT),
        n_variants=3,
    )
    variants = translator.translate(QUESTION, mode="hybrid")
    assert len(variants) >= 2, f"expected at least two variants, got {variants}"
    assert all(variant.strip() for variant in variants), f"empty variant: {variants}"
    return {"variants": variants}


def case_grading() -> dict[str, Any]:
    config = base_config(grading_mode="llm", max_grade_docs=2)
    result = utilis.grade_documents(
        {
            "question": QUESTION,
            "documents": [relevant_document(), irrelevant_document()],
            "web_search_enabled": False,
        },
        {"configurable": {"rag_config": config}},
    )
    kept = result["documents"]
    assert len(kept) == 1, f"expected one relevant document, got {len(kept)}"
    assert "Paris" in kept[0].page_content, kept[0].page_content
    return {
        "kept_sources": [document.metadata.get("source") for document in kept],
        "web_search_used": result["web_search_used"],
    }


def case_generation() -> dict[str, Any]:
    config = base_config()
    documents = utilis._assign_citation_ids(
        [
            relevant_document(),
            Document(
                page_content="France is a country in Western Europe.",
                metadata={"source": "europe.md", "page": 1, "chunk_index": 0},
            ),
        ]
    )
    result = utilis.generate(
        {"question": QUESTION, "documents": documents},
        {"configurable": {"rag_config": config}},
    )
    answer = result["generation"]
    citations = result["citations"]
    assert answer.strip(), "empty answer"
    assert "[S" in answer, f"answer did not contain a citation: {answer}"
    assert citations, "no citations extracted"
    return {
        "answer_preview": answer[:300],
        "citation_ids": [citation["citation_id"] for citation in citations],
        "model_usage": result["usage"],
        "timings_ms": result["timings"],
    }


def case_fast_graph() -> dict[str, Any]:
    config = base_config(
        profile="fast",
        translation_mode="none",
        grading_mode="none",
        enable_web_search=False,
        top_k=1,
    )
    state = run_rag_agent(
        StaticRetriever([relevant_document()]),
        QUESTION,
        enable_web_search=False,
        rag_config=config,
    )
    assert state["generation"].strip(), "empty answer"
    assert state["citations"], "no citations extracted"
    assert state["usage"].get("total_tokens", 0) > 0, "missing token usage"
    return {
        "answer_preview": state["generation"][:300],
        "citation_ids": [citation["citation_id"] for citation in state["citations"]],
        "model_usage": state["usage"],
        "timings_ms": state["timings"],
    }


def case_balanced_graph() -> dict[str, Any]:
    config = base_config(
        profile="balanced",
        translation_mode="multiquery",
        grading_mode="llm",
        max_grade_docs=2,
        enable_web_search=False,
        top_k=2,
    )
    state = run_rag_agent(
        StaticRetriever([relevant_document(), irrelevant_document()]),
        QUESTION,
        enable_web_search=False,
        rag_config=config,
    )
    assert len(state["documents"]) == 1, f"expected one relevant document, got {len(state['documents'])}"
    assert state["generation"].strip(), "empty answer"
    assert state["citations"], "no citations extracted"
    return {
        "answer_preview": state["generation"][:300],
        "citation_ids": [citation["citation_id"] for citation in state["citations"]],
        "model_usage": state["usage"],
        "timings_ms": state["timings"],
    }


CASES = [
    ("translation", case_translation),
    ("grading", case_grading),
    ("generation", case_generation),
    ("fast_graph", case_fast_graph),
    ("balanced_graph", case_balanced_graph),
]
CASE_NAMES = [name for name, _ in CASES]


def _callback_usage(callback) -> dict[str, Any]:
    return {
        "prompt_tokens": callback.prompt_tokens,
        "completion_tokens": callback.completion_tokens,
        "total_tokens": callback.total_tokens,
        "calls": callback.successful_requests,
        "estimated_cost_usd": round(callback.total_cost, 6),
    }


def run_case(name: str) -> CaseResult:
    functions = dict(CASES)
    if name not in functions:
        raise ValueError(f"Unknown real-LLM case: {name}")

    started = time.perf_counter()
    try:
        with get_openai_callback() as callback:
            details = functions[name]()
        usage = _callback_usage(callback)
        if not usage["total_tokens"] and details.get("model_usage"):
            usage = details["model_usage"]
        return CaseResult(
            name=name,
            passed=True,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            usage=usage,
            details=details,
        )
    except Exception as error:
        return CaseResult(
            name=name,
            passed=False,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            error=f"{type(error).__name__}: {error}",
        )


def run_all_cases(limit: int | None = None) -> list[CaseResult]:
    selected = CASES[:limit] if limit else CASES
    return [run_case(name) for name, _ in selected]


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    latencies = [result.latency_ms for result in results]
    return {
        "total": len(results),
        "passed": sum(result.passed for result in results),
        "failed": sum(not result.passed for result in results),
        "total_calls": sum(result.usage.get("calls", 0) for result in results),
        "total_tokens": sum(result.usage.get("total_tokens", 0) for result in results),
        "p50_latency_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
    }
