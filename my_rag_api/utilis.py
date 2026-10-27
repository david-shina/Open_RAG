import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from tavily import TavilyClient
from typing_extensions import TypedDict

from rag_demo.embeddings import Embedding
from rag_demo.loaders import docxloader, pdf_loader, pptx_loader
from rag_demo.prompts import GradeDocument, gradingPrompt, rag_prompt, re_write_prompt
from rag_demo.queryTranslator import QueryTranslator

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_TRANSLATOR_MODEL = "openai/gpt-oss-20b"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


PROFILE_DEFAULTS = {
    "fast": {
        "translation_mode": "none",
        "grading_mode": "none",
        "max_grade_docs": 0,
        "top_k": 5,
        "enable_web_search": False,
    },
    "balanced": {
        "translation_mode": "multiquery",
        "grading_mode": "llm",
        "max_grade_docs": 8,
        "top_k": 5,
        "enable_web_search": True,
    },
    "quality": {
        "translation_mode": "hybrid",
        "grading_mode": "llm",
        "max_grade_docs": 8,
        "top_k": 8,
        "enable_web_search": True,
    },
}


@dataclass(frozen=True)
class RAGConfig:
    profile: str = "balanced"
    top_k: int = 5
    translation_mode: str = "multiquery"
    grading_mode: str = "llm"
    max_grade_docs: int = 8
    enable_web_search: bool = True
    parallel_retrieval: bool = True
    max_workers: int = 4
    llm_model: str = DEFAULT_MODEL
    translator_model: str = DEFAULT_TRANSLATOR_MODEL
    temperature: float = 0.0
    request_timeout: float = 30.0
    n_query_variants: int = 3
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0

    @classmethod
    def from_env(cls, profile: Optional[str] = None) -> "RAGConfig": # This is a factory method that creates a RAGConfig instance based on environment variables and optional profile input.
        requested = (profile or os.getenv("RAG_PROFILE", "balanced")).strip().lower()
        profile_name = requested if requested in PROFILE_DEFAULTS else "balanced"
        defaults = PROFILE_DEFAULTS[profile_name]
        translation_mode = os.getenv(
            "RAG_TRANSLATION_MODE", defaults["translation_mode"]
        ).strip().lower()
        if translation_mode not in {"none", "multiquery", "stepback", "hybrid"}:
            translation_mode = "multiquery"
        grading_mode = os.getenv("RAG_GRADING_MODE", defaults["grading_mode"]).strip().lower()
        if grading_mode not in {"none", "llm"}:
            grading_mode = "llm"

        return cls(
            profile=profile_name,
            top_k=int(os.getenv("RAG_TOP_K", defaults["top_k"])),
            translation_mode=translation_mode,
            grading_mode=grading_mode,
            max_grade_docs=int(os.getenv("RAG_MAX_GRADE_DOCS", defaults["max_grade_docs"])),
            enable_web_search=_env_bool("RAG_WEB_SEARCH", defaults["enable_web_search"]),
            parallel_retrieval=_env_bool("RAG_PARALLEL_RETRIEVAL", True),
            max_workers=int(os.getenv("RAG_MAX_WORKERS", "4")),
            llm_model=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
            translator_model=os.getenv("GROQ_TRANSLATOR_MODEL", DEFAULT_TRANSLATOR_MODEL),
            temperature=float(os.getenv("GROQ_TEMPERATURE", "0")),
            request_timeout=float(os.getenv("RAG_REQUEST_TIMEOUT", "30")),
            n_query_variants=int(os.getenv("N_QUERY_VARIANTS", "3")),
            input_cost_per_million=float(os.getenv("GROQ_INPUT_COST_PER_1M", "0")),
            output_cost_per_million=float(os.getenv("GROQ_OUTPUT_COST_PER_1M", "0")),
        )


def _configure_langsmith() -> None:
    key = os.getenv("LANGCHAIN_API_KEY")
    if not key:
        return
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "open-rag"))
    os.environ.setdefault("LANGCHAIN_API_KEY", key)
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")


_configure_langsmith()


@lru_cache(maxsize=16)
def _chat_model(model: str, temperature: float, timeout: float) -> ChatGroq:
    return ChatGroq(model=model, temperature=temperature, timeout=timeout)


def _llm(config: RAGConfig) -> ChatGroq:
    return _chat_model(config.llm_model, config.temperature, config.request_timeout)


def _translator_llm(config: RAGConfig) -> ChatGroq:
    return _chat_model(config.translator_model, 0.7, config.request_timeout)


@lru_cache(maxsize=8)
def _grader_chain(model: str, temperature: float, timeout: float):
    structured = _chat_model(model, temperature, timeout).with_structured_output(GradeDocument)
    return gradingPrompt | structured


@lru_cache(maxsize=8)
def _rewriter_chain(model: str, temperature: float, timeout: float):
    return re_write_prompt | _chat_model(model, temperature, timeout) | StrOutputParser()


@lru_cache(maxsize=8)
def _query_translator(model: str, n_variants: int, timeout: float) -> QueryTranslator:
    return QueryTranslator(
        llm=_chat_model(model, 0.7, timeout),
        n_variants=n_variants,
    )


@lru_cache(maxsize=1)
def _web_search_tool():
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return None
    return TavilyClient(api_key=key)


def _get_config(config: Optional[RunnableConfig]) -> RAGConfig:
    configured = (config or {}).get("configurable", {}).get("rag_config")
    if isinstance(configured, RAGConfig):
        return configured
    return RAGConfig.from_env()


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def loadingDocument(filePath: str) -> List[Document]:
    extension = Path(filePath).suffix.lower()

    try:
        if extension == ".pdf":
            chunks = pdf_loader.PDFLoader(filePath).load()
        elif extension == ".docx":
            chunks = docxloader.DocxLoader(filePath).load()
        elif extension == ".pptx":
            chunks = pptx_loader.PptxLoader(filePath).load()
        else:
            raise ValueError(
                f"Unsupported file type: {extension}. Supported types: .pdf, .docx, .pptx"
            )

        if not chunks:
            raise ValueError(f"No text could be extracted from file: {filePath}")

        return chunks
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"Error loading file {filePath}: {error}") from error


def embedding(
    filePath: str,
    chunks: List[Document],
    *,
    user_id: str = "1",
    document_id: Optional[str] = None,
    knn: int = 5,
):
    result = Embedding(filePath, chunks).create_or_load_vectorstore(
        userId=user_id,
        documentId=document_id,
        knn=knn,
    )
    return result["retriever"]


class State(TypedDict, total=False):
    question: str
    original_question: str
    generated_queries: List[str]
    web_search_enabled: bool
    web_search_used: bool
    query_rewritten: bool
    retrieved_documents: List[Document]
    documents: List[Document]
    generation: str
    sources: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    timings: Annotated[Dict[str, float], lambda left, right: {**(left or {}), **(right or {})}]
    usage: Annotated[Dict[str, Any], lambda left, right: {**(left or {}), **(right or {})}]


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _deduplicate_documents(documents: List[Document]) -> List[Document]:
    seen = set()
    unique: List[Document] = []
    for document in documents:
        key = _normalize(document.page_content)[:500]
        if key in seen:
            continue
        seen.add(key)
        unique.append(document)
    return unique


def _assign_citation_ids(documents: List[Document]) -> List[Document]:
    for index, document in enumerate(documents, start=1):
        document.metadata = dict(document.metadata or {})
        document.metadata["citation_id"] = f"S{index}"
    return documents


def _document_source(document: Document) -> Dict[str, Any]:
    metadata = document.metadata or {}
    source = (
        metadata.get("source")
        or metadata.get("sourceId")
        or metadata.get("url")
        or metadata.get("file_type")
        or "unknown"
    )
    return {
        "citation_id": metadata.get("citation_id"),
        "source": source,
        "page": metadata.get("page"),
        "chunk_index": metadata.get("chunk_index"),
        "url": metadata.get("url"),
        "snippet": document.page_content[:200],
    }


def _extract_sources(documents: List[Document]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    seen = set()
    for document in documents:
        source = _document_source(document)
        key = (
            str(source["source"]),
            source["page"],
            source["chunk_index"],
            source["citation_id"],
        )
        if key in seen:
            continue
        seen.add(key)
        sources.append(source)
    return sources


def _format_docs(documents: List[Document]) -> str:
    return "\n\n".join(
        f"[{document.metadata.get('citation_id', 'S?')}] {document.page_content}"
        for document in documents
    )


_CITATION_TRANSLATION = str.maketrans(
    {
        "\u3010": "[",
        "\u3011": "]",
        "\uff3b": "[",
        "\uff3d": "]",
    }
)


def _normalize_citations(text: str) -> str:
    return (text or "").translate(_CITATION_TRANSLATION)


def _extract_citations(answer: str, documents: List[Document]) -> List[Dict[str, Any]]:
    by_id = {
        document.metadata.get("citation_id"): document
        for document in documents
        if document.metadata.get("citation_id")
    }
    cited: List[Dict[str, Any]] = []
    seen = set()
    for group in re.findall(r"\[([^\]]+)\]", _normalize_citations(answer)):
        for match in re.finditer(r"S(\d+)", group):
            citation_id = f"S{match.group(1)}"
            if citation_id in seen or citation_id not in by_id:
                continue
            seen.add(citation_id)
            cited.append(_document_source(by_id[citation_id]))
    return cited


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def _usage_from_response(response: Any, config: RAGConfig) -> Dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        usage = (getattr(response, "response_metadata", None) or {}).get("token_usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    completion_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
    cost = (
        prompt_tokens / 1_000_000 * config.input_cost_per_million
        + completion_tokens / 1_000_000 * config.output_cost_per_million
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(cost, 6),
    }


def retrieve(state: State, config: RunnableConfig) -> Dict[str, Any]:
    started = time.perf_counter()
    question = state["question"]
    rag_config = _get_config(config)
    retriever = config.get("configurable", {}).get("retriever")

    if not retriever:
        logger.warning("No retriever found in config; returning empty results")
        return {
            "question": question,
            "generated_queries": [question],
            "retrieved_documents": [],
            "documents": [],
            "sources": [],
            "timings": {"retrieve": _elapsed_ms(started)},
        }

    queries = [question]
    if rag_config.translation_mode != "none":
        try:
            translator = _query_translator(
                rag_config.translator_model,
                rag_config.n_query_variants,
                rag_config.request_timeout,
            )
            variants = translator.translate(question, mode=rag_config.translation_mode)
            queries = list(dict.fromkeys([question] + variants))
        except Exception as error:
            logger.warning("Query translation failed, falling back to original query: %s", error)

    merged: List[Document] = []
    if rag_config.parallel_retrieval and len(queries) > 1:
        with ThreadPoolExecutor(max_workers=rag_config.max_workers) as executor:
            futures = [executor.submit(retriever.invoke, query) for query in queries]
            for query, future in zip(queries, futures):
                try:
                    merged.extend(future.result())
                except Exception as error:
                    logger.warning("Retrieval failed for query '%s': %s", query, error)
    else:
        for query in queries:
            try:
                merged.extend(retriever.invoke(query))
            except Exception as error:
                logger.warning("Retrieval failed for query '%s': %s", query, error)

    merged = _deduplicate_documents(merged)
    merged = _assign_citation_ids(merged)
    return {
        "question": question,
        "generated_queries": queries,
        "retrieved_documents": merged,
        "documents": merged,
        "sources": _extract_sources(merged),
        "timings": {"retrieve": _elapsed_ms(started)},
    }


def grade_documents(state: State, config: RunnableConfig) -> Dict[str, Any]:
    started = time.perf_counter()
    question = state["question"]
    documents = list(state.get("documents", []))
    rag_config = _get_config(config)
    web_search_enabled = state.get("web_search_enabled", rag_config.enable_web_search)

    broad_phrases = [
        "what is this paper about",
        "summarize",
        "overview",
        "abstract",
        "give me a summary",
        "introduction to",
    ]

    if not documents:
        return {
            "documents": [],
            "sources": [],
            "question": question,
            "web_search_used": bool(web_search_enabled),
            "timings": {"grade_documents": _elapsed_ms(started)},
        }

    if any(phrase in question.lower() for phrase in broad_phrases):
        documents = _assign_citation_ids(documents)
        return {
            "documents": documents,
            "sources": _extract_sources(documents),
            "question": question,
            "web_search_used": False,
            "timings": {"grade_documents": _elapsed_ms(started)},
        }

    filtered: List[Document] = []
    if rag_config.grading_mode != "none":
        limit = rag_config.max_grade_docs if rag_config.max_grade_docs > 0 else len(documents)
        grader = _grader_chain(
            rag_config.llm_model,
            rag_config.temperature,
            rag_config.request_timeout,
        )
        for document in documents[:limit]:
            try:
                score = grader.invoke({"question": question, "document": document.page_content})
                grade = str(score.binary_score).strip().lower()
            except Exception as error:
                logger.warning("Grading failed, keeping document: %s", error)
                filtered.append(document)
                continue

            if grade == "yes":
                filtered.append(document)
    else:
        filtered = documents

    filtered = _assign_citation_ids(filtered)
    need_web = bool(web_search_enabled and not filtered)
    return {
        "documents": filtered,
        "sources": _extract_sources(filtered),
        "question": question,
        "web_search_used": need_web,
        "timings": {"grade_documents": _elapsed_ms(started)},
    }


def decide_to_generate(state: State) -> str:
    if state.get("web_search_used"):
        logger.info("---DECISION: TRANSFORM QUERY (web search fallback)---")
        return "transform_query"
    logger.info("---DECISION: GENERATE---")
    return "generate"


def transform_query(state: State, config: RunnableConfig) -> Dict[str, Any]:
    started = time.perf_counter()
    question = state["question"]
    documents = state.get("documents", [])
    rag_config = _get_config(config)

    try:
        rewriter = _rewriter_chain(
            rag_config.llm_model,
            rag_config.temperature,
            rag_config.request_timeout,
        )
        better_question = rewriter.invoke({"question": question}).strip()
    except Exception as error:
        logger.warning("Query rewrite failed, keeping original: %s", error)
        better_question = question

    return {
        "documents": documents,
        "question": better_question,
        "query_rewritten": True,
        "timings": {"transform_query": _elapsed_ms(started)},
    }


def web_search(state: State, config: RunnableConfig) -> Dict[str, Any]:
    started = time.perf_counter()
    question = state["question"]
    documents = list(state.get("documents", []))

    tool = _web_search_tool()
    if tool is None:
        logger.error("Web search requested but TAVILY_API_KEY is not configured")
        return {
            "question": question,
            "documents": documents,
            "sources": _extract_sources(documents),
            "web_search_used": False,
            "timings": {"web_search": _elapsed_ms(started)},
        }

    try:
        result = tool.search(query=question, max_results=5)
        result_list = result.get("results", [])
    except Exception as error:
        logger.error("Web search failed: %s", error)
        return {
            "question": question,
            "documents": documents,
            "sources": _extract_sources(documents),
            "web_search_used": False,
            "timings": {"web_search": _elapsed_ms(started)},
        }

    web_documents: List[Document] = []
    for item in result_list:
        url = item.get("url", "")
        content = item.get("content", "")
        if not content:
            continue
        web_documents.append(
            Document(
                page_content=f"Source: {url}\nContent: {content}",
                metadata={"source": url, "url": url},
            )
        )

    combined = _deduplicate_documents(documents + web_documents)
    combined = _assign_citation_ids(combined)
    return {
        "question": question,
        "documents": combined,
        "sources": _extract_sources(combined),
        "web_search_used": True,
        "timings": {"web_search": _elapsed_ms(started)},
    }


def generate(state: State, config: RunnableConfig) -> Dict[str, Any]:
    started = time.perf_counter()
    question = state["question"]
    documents = _assign_citation_ids(list(state.get("documents", [])))
    rag_config = _get_config(config)
    sources = _extract_sources(documents)

    if not documents:
        generation = "I couldn't find any relevant information to answer your question."
        return {
            "generation": generation,
            "sources": sources,
            "citations": [],
            "usage": {},
            "timings": {"generate": _elapsed_ms(started)},
        }

    try:
        messages = rag_prompt.format_messages(question=question, context=_format_docs(documents))
        response = _llm(rag_config).invoke(messages)
        generation = _normalize_citations(_message_text(response))
        usage = _usage_from_response(response, rag_config)
    except Exception as error:
        logger.error("LLM generation failed: %s", error)
        generation = (
            "An error occurred while generating an answer. "
            "Please ensure your API keys are configured."
        )
        usage = {}

    citations = _extract_citations(generation, documents)
    return {
        "generation": generation,
        "sources": sources,
        "citations": citations,
        "usage": usage,
        "timings": {"generate": _elapsed_ms(started)},
    }


def graph_builder():
    workflow = StateGraph(state_schema=State)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"transform_query": "transform_query", "generate": "generate"},
    )
    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


app = graph_builder()


def run_rag_agent(
    retriever,
    question: str,
    enable_web_search: Optional[bool] = None,
    rag_config: Optional[RAGConfig] = None,
) -> Dict[str, Any]:
    config = rag_config or RAGConfig.from_env()
    if enable_web_search is not None:
        config = replace(config, enable_web_search=bool(enable_web_search))
    inputs = {
        "question": question,
        "original_question": question,
        "web_search_enabled": config.enable_web_search,
    }
    run_config = {"configurable": {"retriever": retriever, "rag_config": config}}
    return app.invoke(inputs, config=run_config)


def chat(
    path: str,
    user_input: str,
    enable_web_search: Optional[bool] = None,
    rag_config: Optional[RAGConfig] = None,
) -> str:
    if not Path(path).exists():
        logger.error("File path does not exist")
        return "Error: File path does not exist."

    config = rag_config or RAGConfig.from_env()
    chunks = loadingDocument(path)
    result = Embedding(path, chunks).create_or_load_vectorstore(
        userId="1",
        documentId=Path(path).stem,
        knn=config.top_k,
    )

    final_state = run_rag_agent(
        result["retriever"],
        user_input,
        enable_web_search=enable_web_search,
        rag_config=config,
    )
    return final_state.get("generation", "Error: No generation produced.")
