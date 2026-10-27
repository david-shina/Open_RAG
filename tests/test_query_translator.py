import pytest
from langchain_core.runnables import RunnableLambda

from rag_demo.queryTranslator import QueryTranslator


def _translator(response):
    return QueryTranslator(llm=RunnableLambda(lambda _: response), n_variants=3)


def test_multiquery_cleans_numbered_lines():
    translator = _translator("1. First question?\n2. Second question?\n- Third question?")
    assert translator.generate_multiquery("original") == [
        "First question?",
        "Second question?",
        "Third question?",
    ]


def test_translate_stepback_and_hybrid():
    stepback = _translator("broader question")
    assert stepback.translate("narrow question", mode="stepback") == ["broader question"]

    def hybrid_response(prompt_value):
        if "step back" in str(prompt_value).lower():
            return "broader question"
        return "1. variant one\n2. variant two\n3. broader question"

    hybrid = QueryTranslator(llm=RunnableLambda(hybrid_response), n_variants=3)
    assert hybrid.translate("question", mode="hybrid") == [
        "variant one",
        "variant two",
        "broader question",
    ]


def test_translate_rejects_unknown_mode():
    translator = _translator("anything")
    with pytest.raises(ValueError):
        translator.translate("question", mode="unknown")
