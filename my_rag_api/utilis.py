import logging
from langchain_groq import ChatGroq
from typing import List
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph, END
from rag_demo.loaders import pdf_loader, pptx_loader, docxloader, base_loader
from rag_demo.embeddings import Embedding
from rag_demo.prompts import *
from rag_demo.queryTranslator import QueryTranslator
from pathlib import Path
from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from tavily import TavilyClient
from dotenv import load_dotenv
import os
from pprint import pprint
from typing_extensions import TypedDict


load_dotenv()
langchainApiKey = os.getenv('LANGCHAIN_API_KEY')
groqApiKey = os.getenv('GROQ_API_KEY')
braveApiKey = os.getenv('BRAVE_API_KEY')
tavilyApiKey = os.getenv('TAVILY_API_KEY')

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'default'
os.environ['LANGCHAIN_API_KEY'] = langchainApiKey
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['GROQ_API_KEY']= groqApiKey
os.environ['TAVILY_API_KEY'] = tavilyApiKey
client = Client()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.7)
structured_llm_output = llm.with_structured_output(GradeDocument)
qt = QueryTranslator(llm=llm)


##########################PROMPTS###################################
rag_prompt = client.pull_prompt('rlm/rag-prompt')

def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

rag_chain = rag_prompt | llm | StrOutputParser()

grader_chain = gradingPrompt | structured_llm_output

web_search_tool = TavilyClient()

question_rewriter = re_write_prompt | llm | StrOutputParser()

##Loading of Documents##
def loadingDocument(filePath):
    try:
        extension = Path(filePath).suffix.lower()

        if extension == '.pdf':
            chunks = pdf_loader.PDFLoader(filePath).load()

        elif extension == '.docx':
            chunks = docxloader.DocxLoader(filePath).load()
        
        elif extension == '.pptx':
            chunks = pptx_loader.PptxLoader(filePath).load()

        else:
            print(f'Invalid file type: {extension}')

        if not chunks:
            raise ValueError(f'No content could be loaded from file: {filePath}')

        return chunks
    except ValueError:
        raise f'Unsupported file type: {extension}. Supported types: .pdf, .docx, .pptx'

    except Exception as e:
        raise ValueError(f'Error loading file {filePath}: {str(e)}')
    



def embedding(filePath, chunks):
    retreiver, _ = Embedding(filePath, chunks).create_or_load_vectorstore(userId='1')

    return retreiver

def queryTranslator(query:str):
    return qt.translate(query, mode='hybrid')




class State(TypedDict):
    question: str
    filePath: str
    web_search: bool = False
    documents: List[Document]
    generation: str

_vector_store_cache = {}
def retrieve(state: State, config):
    
    question = state['question']
    # Try getting it from config (works for local 'chat' function)
    retriever = config.get('configurable', {}).get('retriever')
    
    # Fallback logic if running in Studio (where config retriever is None)
    if not retriever:
        # You might need a default path or logic to load a store here
        # For now, just logging a warning so it doesn't crash immediately
        logger.warning("No retriever found in config! using fallback or empty results")
        return {'documents': [], 'question': question}
    documents = retriever.invoke(question)

    if not documents:
        print("---WARNING: RETRIEVER FOUND 0 DOCUMENTS---")
    else:
        print(f"---RETRIEVED {len(documents)} DOCUMENTS---")

  

    return {'documents':documents, 'question':question}

def generate(state: State):
    question = state['question']
    documents = state['documents']

    generation = rag_chain.invoke({'question':question, 'context':documents})

    return {'documents':documents, 'question':question, 'generation':generation}

def grade_documents(state:State):
    question = state['question']
    documents = state['documents']
    filteredDocs = []
    web_search = False

    broad_phrases = [
        "what is this paper about",
        "summarize",
        "overview",
        "abstract",
        "give me a summary",
        "introduction to"
    ]

    if any(p in question.lower() for p in broad_phrases):
        return {'documents':documents, 'question':question, 'web_search':False}
    
    for doc in documents:
        score = grader_chain.invoke({'question': question, 'document': doc.page_content})
        grade = score.binary_score

        print(f"Doc snippet: {doc.page_content[:60]}...")
        print(f"--> Grade: {grade}")

        if grade.lower() == 'yes':
            filteredDocs.append(doc)
        else:
            web_search = True
            continue

    return {"documents": filteredDocs, 'question': question, 'web_search': web_search}
def transform_query(state: State):

    question = state['question']
    documents = state['documents']

    better_question = question_rewriter.invoke({'question':question})

    return {'documents':documents, 'question':better_question}

def web_search(state: State):

    question = state['question']
    documents = state['documents']

    if not web_search_tool:
        logger.error('Web search attempted but tool is unavailable.')
        return {'question': question, 'documents': documents}

    result = web_search_tool.search(query=question)

    resultList = result.get('results', [])

    webContent = "\n\n".join(
        [f"Source: {d.get('url')}\nContent: {d.get('content')}" for d in resultList]
    )

    if webContent:
        web_results = Document(page_content=webContent)

        documents.append(web_results)

    return {'question':question, 'documents':documents}


def decide_to_generate(state: State):

    state['question']
    web_search = state['web_search']
    docs = state['documents']

    if web_search == True:
        logger.info(
            f"---DECISION:  TRANSFORM QUERY---. Number of Docs: {len(docs)} relevant docs"
        )

        return 'transform_query'
    
    else:
        logger.info("---DECISION: GENERATE---")
        return 'generate'


def graphBuilder():

    workflow = StateGraph(state_schema=State)

    workflow.add_node('retrieve', retrieve)
    workflow.add_node('grade_documents', grade_documents)
    workflow.add_node('generate', generate)
    workflow.add_node('transform_query', transform_query)
    workflow.add_node('web_search', web_search)

    workflow.add_edge(START, 'retrieve')
    workflow.add_edge('retrieve', 'grade_documents')
    workflow.add_conditional_edges(
        'grade_documents',
        decide_to_generate,
        {
            'transform_query': 'transform_query',
            'generate':'generate'
        }
    )
    workflow.add_edge('transform_query', 'web_search')
    workflow.add_edge('web_search', 'generate')
    workflow.add_edge('generate', END)


    return workflow.compile()


app = graphBuilder()


def chat(path, user_input:str):

    if not Path(path).exists():
        logger.error('File path does not exist')
        return 'Error: File path does not exist.'

    chunks = loadingDocument(path)

    retriever= embedding(path,chunks)


    inputs = {'question':user_input}

    for output in app.stream(inputs, config={'configurable':{'retriever':retriever}}):
        for key, value in output.items():
            pprint(f"Node '{key}':")
            final_value = value
        pprint("\n---\n")

    if final_value and isinstance(final_value, dict):
        result = final_value.get('generation', '⚠️ No generation produced')
        return result
    else:
        logger.error('No final state produced by the graph.')
        return 'Error: No final state produced.'
    

