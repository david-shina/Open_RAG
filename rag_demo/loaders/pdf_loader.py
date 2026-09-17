import re
import fitz
import logging
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.utils.constants import PartitionStrategy
from langchain_core.documents import Document
from . import base_loader 
import pdf2image
from typing import List, Generator
from concurrent.futures import ThreadPoolExecutor
#from django.conf import settings



load_dotenv(dotenv_path='.env')
langchainApiKey = os.getenv('LANGCHAIN_API_KEY')

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'default'
os.environ['LANGCHAIN_API_KEY'] = langchainApiKey
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MIN_TEXT_REQ = 30

class PDFLoader(base_loader.BaseLoader):

    def __init__(self, filePath:str, chunk_size:int=500, chunk_overlap:int=50):
        super().__init__(filePath)
        self.chunk_size = chunk_size
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

        logger.info(f'Loading text from: {self.file_path}')
    

    def detect_type(self, samplePages:int = 3)->str:
        try:
            with fitz.open(self.file_path) as docs:

                for page in docs[:samplePages]:
                    text = page.get_text().strip()
                    if len(text) > MIN_TEXT_REQ:
                        return 'Text-based PDF'
                
        except Exception as e:
            logger.error(f"Error reading text-based PDF: {e}")


        try:
            images = pdf2image.convert_from_path(
                self.file_path, 
                first_page=1, 
                last_page=1)
            if images:
                return "Image-based PDF"
        except Exception as e:
            logger.error(f'Error reading image-based PDF: {e}')

        return 'Unknown PDF type'
    
    def _extract_text_pdf(self)->Generator[Document, None, None]:
        try:
            loader = PyPDFLoader(self.file_path)
            allPages = loader.load()
            foundFirstPage = False

            if not allPages:
                raise ValueError('PyPDFLoader returned no pages')

            for page in allPages:

                if not foundFirstPage and any(p.search(page.page_content) for p in self.start_patterns):
                    foundFirstPage=True

                if foundFirstPage:
                    yield page

            if not foundFirstPage:
                yield from allPages

            logger.info("PyPDFLoader succeeded")
            return
        
        except Exception as e:
            print(f"PyPDFLoader failed: {e}. Falling back to PyMuPDF...")


        try:
            pdfDoc = fitz.open(self.file_path)
            foundFirstPage = False

            for page_num in range(pdfDoc.page_count):
                page = pdfDoc[page_num]
                content = page.get_text()

                if not content.strip():
                    continue

                if not foundFirstPage and any(p.search(content) for p in self.start_patterns):
                    foundFirstPage=True
                
                if foundFirstPage:
                    document = Document(
                     page_content=content,
                     metadata = {
                         'source':self.file_path,
                         'page': page_num
                    }
                    )

                    yield document
            if not foundFirstPage:
                for page_num in range(pdfDoc.page_count):
                    page = pdfDoc[page_num]
                    content = page.get_text()

                    if content.strip():
                        document = Document(
                            page_content=content,
                            metadata = {
                         'source':self.file_path,
                         'page': page_num
                        }
                        )

                        yield document
            
            pdfDoc.close()

        except Exception as e:
            raise RuntimeError(f"Both PyPDFLoader and PyMuPDF failed to extract text: {e}")


    def _process_image_page(self, page)->Document:

        return page
    
    def _extract_image_pdf(self)->Generator[Document, None, None]:
        try:
            loader = UnstructuredLoader(
                file_path=self.file_path, 
                strategy=PartitionStrategy.HI_RES)
            all_pages = loader.load()
            if not all_pages:
                raise ValueError("Unstructured returned no pages")

            for page in all_pages:
                yield page

        except Exception as e:
            raise RuntimeError(f"OCR extraction failed: {e}")


    
    def load(self)->List[Document]:
        doc_type = self.detect_type()

        if doc_type == 'Text-based PDF':
            pages_gen = self._extract_text_pdf()
        elif doc_type == "Image-based PDF":
            pages_gen = self._extract_image_pdf()
        else:
            logger.error(f"Unsupported or unreadable PDF: {self.file_path}")
            return []
        
        pages = list(pages_gen)
        if not pages:
            logger.warning(f"No text extracted from: {self.file_path}")
            return []
        

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap
        )

        docs_chunks = splitter.split_documents(pages)

        for i, chunk in enumerate(docs_chunks):
            chunk.metadata.update({
                "doc_id": os.path.basename(self.file_path),
                "chunk_index": i,
                "source_type": doc_type
            })

        logger.info(f"Loaded {len(docs_chunks)} chunks from {self.file_path}")
        return docs_chunks
    

# if __name__ == '__main__':
#     filePath = "C:/Users/Dell/Documents/COR.pdf"
#     loader = PDFLoader(filePath)

#     chunks = loader.load()
#     print(loader.detect_type())
    
#     if chunks:
#         print('Successful')
#         print(chunks)
#     else:
#         print('Check Errors')



