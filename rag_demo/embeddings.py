import os
import re
from langchain_ollama import OllamaEmbeddings
import logging
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
from pathlib import Path
from joblib import Memory

from typing import List, Optional
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
load_dotenv('.env')
langchainApiKey = os.getenv('LANGCHAIN_API_KEY')

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'default'
os.environ['LANGCHAIN_API_KEY'] = langchainApiKey
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BatchedEmbeddingWrapper:

    def __init__(self, parent):
        self.parent = parent

    def embed_query(self, text:str):
        return self.parent.compute_embedding(text)
    
    def embed_documents(self, texts:List[str]):
        string_texts = [str(t) for t in texts]
        return self.parent.embed_document_in_batches(string_texts)



class Embedding:
    def __init__(self,  path:str ,chunks:List[Document]):
        self.embeddingModel = OllamaEmbeddings(model='nomic-embed-text:latest')
        self.path = path
        self.chunks = chunks
    
    def set_embedding_model(self, model_type:str, model_name:str, **kwargs):
        
        # How to dynamically set the embedding model
        try:
            model_type = model_type.lower()
            if model_type == 'ollama':
                self.embeddingModel = OllamaEmbeddings(model=model_name or 'nomic-embed-text', **kwargs)
            elif model_type == 'huggingface':
                self.embeddingModel = HuggingFaceEmbeddings(
                    model = model_name or 'sentence-transformers/all-MiniLM-L6-v2',
                    model_kwargs = {'device':'cpu'},
                    encode_kwargs = {"normalize_embeddings": False},
                    cache_folder = './RAG')
            # elif model_type == 'openai':
            #     self.embeddingModel = OpenAIEmbeddings(model=model_name or 'text-embedding-ada-002', **kwargs)
            else:    
                raise ValueError(f'Unsupported embedding model type: {model_type}. Supported model_types are openai, ollama and huggingface, support for more models incoming')
            
            test_embed = self.embeddingModel.embed_query('Test Document')
            logger.info(f'Switched to {model_type} embedding model, dimension: {len(test_embed)}')
        
        except Exception as e:
            logger.error(f"Failed to set embedding model: {str(e)}")
            raise

    def _ensure_model(self):
        if self.embeddingModel is None:
            raise RuntimeError("Embedding model not initialized")

    def compute_embedding(self, text:str):
        self._ensure_model()
        return self.embeddingModel.embed_query(text)

    def embed_document_in_batches(self, documents, batch_size=32):
        self._ensure_model()
        all_embeddings = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            texts = [doc for doc in batch]
            
            batch_vectors = self.embeddingModel.embed_documents(texts)
            all_embeddings.extend(batch_vectors)
        return all_embeddings
    
    def deduplicate_chunks(self, doc:List[Document])->List[Document]:  
        pass
    
    def _collection_name(self, user_id: Optional[str] = None) -> str:
        stem = Path(self.path).stem

        stem = re.sub(r"[^a-zA-Z0-9._-]", "_", stem)
        stem = re.sub(r"^[^a-zA-Z0-9]+", "", stem)
        stem = re.sub(r"[^a-zA-Z0-9]+$", "", stem)
        
        if user_id:
            return f"rag_{user_id}_{re.sub(r"[^a-zA-Z0-9._-]", "_", stem)}_vector"
        return f"rag:{re.sub(r"[^a-zA-Z0-9._-]", "_", stem)}"


    def create_or_load_vectorstore(self, userId: Optional[str],saveVector:bool =True ,knn:int=5,**kwargs):
        
        persist_dir = f'./chroma_{Path(self.path).stem}.db'
        collectionName = self._collection_name(userId)
        batched_embed = BatchedEmbeddingWrapper(self)
        try:
            if os.path.exists(persist_dir):
                logger.info(f'Loading existing vector store from {persist_dir}')
                vectorStore = Chroma(
                    persist_directory= persist_dir,
                    collection_name=collectionName,
                    embedding_function=batched_embed
                )
            else:
                vectorStore = Chroma.from_documents(
                    documents=self.chunks,
                    embedding=batched_embed,
                    persist_directory=persist_dir,
                    collection_name=collectionName
                )   
            
            # if saveVector:
            #     vectorStore.persist()

            retriever = vectorStore.as_retriever(search_kwargs = {'k':knn}, **kwargs)

            return {
            "retriever": retriever,          # internal use only
            "collection_name": collectionName,
            "persist_path": str(persist_dir),
            }

            
        
            return vectorStore.as_retriever(search_kwargs = {'k':knn}, **kwargs), persist_dir
        
        except Exception as e:
            logger.error(f'Error in vectorstore creating / loading: {str(e)}')
            raise


    def cleanup_vectorstore(self):
        persist_dir = f'./vectorstores/{Path(self.path).stem}'
        if os.path.exists(persist_dir):
            import shutil
            shutil.rmtree(persist_dir)
            logger.info(f"Vector store at {persist_dir} cleaned up")
        else:
            logger.info(f"No vector store found at {persist_dir}")



# if __name__ == '__main__':

#     loader = pptx_loader.PptxLoader("C:/Users/Dell/Downloads/Project-Proposal (1).pptx")
#     chunks = loader.load()
#     embed = Embedding(path="C:/Users/Dell/Downloads/Project-Proposal (1).pptx", chunks=chunks)

#     try:
#         embed.set_embedding_model(model_type='huggingface', model_name='sentence-transformers/all-MiniLM-L6-v2')
#         embed.create_or_load_vectorstore(userId='1')
#     except Exception as e:
#         print(f'{e}')