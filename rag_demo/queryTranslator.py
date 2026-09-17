import logging
from typing import List, Literal, Dict, Any
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_groq import ChatGroq
from .prompts import multiquery_prompt, stepBackPrompt

load_dotenv(dotenv_path='.env')
langchainApiKey = os.getenv('LANGCHAIN_API_KEY')
groqApiKey = os.getenv('GROQ_API_KEY')

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'default'
os.environ['LANGCHAIN_API_KEY'] = langchainApiKey
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['GROQ_API_KEY'] = groqApiKey


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QueryTranslator:

    def __init__(self, llm = None, n_variants:int = 3):

        self.llm = llm or ChatGroq(model='openai/gpt-oss-20b', temperature=0.7)
        self.n_variants = n_variants

        self.multiquery_prompt = multiquery_prompt

        self.stepback_prompt = stepBackPrompt


    def generate_multiquery(self, query:str) -> List[str]:

        chain = (self.multiquery_prompt|self.llm | StrOutputParser())
        result = chain.invoke({'query':query, 'n':self.n_variants})
        queries = [q.strip() for q in result.split('\n') if q.strip('\n')]
        logger.info(f"[MultiQuery] Generated {len(queries)} queries for: {query}")
        return queries
    
    def generate_stepback(self, query:str) -> str:
        chain = self.stepback_prompt | self.llm | StrOutputParser()
        result = chain.invoke({'question':query}).strip()
        logger.info(f"[Step-back] Original: {query} | Step-back: {result}")
        return result
    
    def translate(self, query:str, mode:Literal['multiquery', 'stepback', 'hybrid']):

        if mode == "multiquery":
            return self.generate_multiquery(query)

        elif mode == "stepback":
            return [self.generate_stepback(query)]

        elif mode == "hybrid":
            variants = self.generate_multiquery(query)
            stepback = self.generate_stepback(query)
            return list(set(variants + [stepback]))

        else:
            raise ValueError(f"Unsupported translation mode: {mode}")
