import os
import re
import logging
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredWordDocumentLoader
from . import base_loader
from dotenv import load_dotenv
from typing import Generator, List
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path='.env')
langchainApiKey = os.getenv('LANGCHAIN_API_KEY')

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'default'
os.environ['LANGCHAIN_API_KEY'] = langchainApiKey
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

class DocxLoader(base_loader.BaseLoader):
    def __init__(self, file_path:str, chunk_size:int=500, chunk_overlap:int=50):
        super().__init__(file_path)
        self.chunk_size = 500
        self.chunk_overlap = chunk_overlap
        self.start_patterns  = [
            re.compile(pattern) for pattern in [
                r'Chapter 1\b',
                r'Chapter One\b',
                r'CHAPTER ONE\b',
                r'CHAPTER 1\b',
                r'1\.\s+INTRODUCTION\b',
                r'1\s+Introduction\b',
                r'1\s+Course Overview\b',
                r'1\.\s+COURSE OVERVIEW\b',
                r'\bTable of Contents\b',
                r'\bContents\b',
                r'\bCONTENTS\b',
                r'\bIndex\b',
                r'\bTOC\b'
            ] 
        ]
        logger.info(f"Initializing loader for: {self.file_path}")


    def detect_type(self):
        return super().detect_type()
    
    def _extract_text(self)->Generator[Document, None, None]:
        loaders = [
            ('Doc2txtLoader', Docx2txtLoader(self.file_path)),
            ('UnstructuredWordDocumentLoader', UnstructuredWordDocumentLoader(self.file_path))
        ]
        for name, loader in loaders:

            try:
                
                logger.info(f"Attempting extraction with {name}")
                allPages = loader.load()
                foundFirstPage = False
                if not allPages:
                    raise ValueError(f"{name} returned no pages")

                for page in allPages:
                    if not foundFirstPage and any(p.search(page.page_content) for p in self.start_patterns):
                        foundFirstPage = True

                    if foundFirstPage:
                        yield page
                    
                if not foundFirstPage:
                    for page in allPages:
                        yield page

                logger.info(f"{name} succeeded")
                return

            except Exception as e:
                logger.warning(f"{name} failed: {e}")

        logger.error(f"All loaders failed for {self.file_path}")


    def load(self)->List[Document]:
        docs = list(self._extract_text())

        if not docs: 
            logger.warning(f'File is empty or extraction of text failed successfully')
            return []
        
        try:
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size = self.chunk_size,
                chunk_overlap = self.chunk_overlap
            )
        except Exception:
            logger.warning("Tiktoken encoder unavailable, falling back to CharacterTextSplitter")
            splitter = CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

        chunks = splitter.split_documents(docs)

        for i, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    'sourceId':os.path.basename(self.file_path),
                    'chunk_index':i,
                    'source_type':'Text-based',
                    'file_type':'docx'
                })
            
        return chunks


# if __name__ == '__main__':
#     filePath = "C:/Users/Dell/Desktop/Project/Proposal.docx"
#     loader = DocxLoader(file_path=filePath)

#     chunks = loader.load()
    
#     if chunks:
#         print('Successful')
#         print(chunks[0])
#     else:
#         print('Check Errors')
