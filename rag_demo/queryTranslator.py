import logging
import os
import re
from typing import List, Literal

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from .prompts import multiquery_prompt, stepBackPrompt

load_dotenv(dotenv_path=".env")
langchainApiKey = os.getenv("LANGCHAIN_API_KEY")
groqApiKey = os.getenv("GROQ_API_KEY")

if langchainApiKey:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "open-rag")
    os.environ["LANGCHAIN_API_KEY"] = langchainApiKey
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
if groqApiKey:
    os.environ["GROQ_API_KEY"] = groqApiKey


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TRANSLATOR_MODEL = "openai/gpt-oss-20b"


def _clean_query(line: str) -> str:
    line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line)
    return line.strip().strip('"')


class QueryTranslator:
    def __init__(self, llm=None, n_variants: int = 3, model: str | None = None, temperature: float = 0.7):
        self.llm = llm or ChatGroq(
            model=model or os.getenv("GROQ_TRANSLATOR_MODEL", DEFAULT_TRANSLATOR_MODEL),
            temperature=temperature,
        )
        self.n_variants = n_variants
        self.multiquery_prompt = multiquery_prompt
        self.stepback_prompt = stepBackPrompt

    def generate_multiquery(self, query: str) -> List[str]:
        chain = self.multiquery_prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": query, "n": self.n_variants})
        queries = [cleaned for line in result.split("\n") if (cleaned := _clean_query(line))]
        queries = list(dict.fromkeys(queries))[: self.n_variants]
        logger.info("[MultiQuery] Generated %s queries for: %s", len(queries), query)
        return queries

    def generate_stepback(self, query: str) -> str:
        chain = self.stepback_prompt | self.llm | StrOutputParser()
        result = chain.invoke({"question": query}).strip()
        logger.info("[Step-back] Original: %s | Step-back: %s", query, result)
        return result

    def translate(self, query: str, mode: Literal["multiquery", "stepback", "hybrid"]):
        if mode == "multiquery":
            return self.generate_multiquery(query)

        if mode == "stepback":
            return [self.generate_stepback(query)]

        if mode == "hybrid":
            variants = self.generate_multiquery(query)
            stepback = self.generate_stepback(query)
            return list(dict.fromkeys(variants + [stepback]))

        raise ValueError(f"Unsupported translation mode: {mode}")
