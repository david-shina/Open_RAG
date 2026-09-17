
import os
import tempfile
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from rag_demo.queryTranslator import QueryTranslator
from rag_demo.embeddings import Embedding
from rag_demo.prompts import *
from rag_demo.loaders import (pdf_loader, docxloader, pptx_loader)
from langchain_groq import ChatGroq


from .serializers import DocumentUploadSerializer

from pathlib import Path



llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.7)
structured_llm_output = llm.with_structured_output(GradeDocument)
qt = QueryTranslator(llm=llm)




def loadingDocument(file_path):
    """Helper function to instantiate the right loader based on extension."""
    try:
        extension = Path(file_path).suffix.lower()


        if extension == '.pdf':
            chunks = pdf_loader.PDFLoader(file_path).load()
        elif extension == '.docx':
            chunks = docxloader.DocxLoader(file_path).load()
        elif extension == '.pptx':
            chunks = pptx_loader.PptxLoader(file_path).load()
        else: 
            raise ValueError(f"No loader available for extension: {extension}")  

        if not chunks:
            raise ValueError("No text could be extracted from the document.")         
        return chunks 
    except ValueError:
        raise f'Unsupported file type: {extension}. Supported types: .pdf, .docx, .pptx'

    except Exception as e:
        raise ValueError(f'Error loading file {file_path}: {str(e)}')
    
      

def embedding(filePath, chunks):
    retreiver, _ = Embedding(filePath, chunks).create_or_load_vectorstore(userId='1')

    return retreiver

def queryTranslator(query:str):
    return qt.translate(query, mode='hybrid') 

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def ingest_document(request):
    """
    Accepts a file upload, detects type, extracts text, and returns chunks.
    """
    # 1. Validate incoming data
    serializer = DocumentUploadSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    uploaded_file = request.FILES['file']
    tmp_file_path = None
    file_type = Path(uploaded_file.name).suffix.lower().strip('.')
    try:
        # 2. Save to temporary file (since loaders need a file path)
        # delete=False because we need to close the write handle before the loader opens it
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            for chunk in uploaded_file.chunks():
                tmp_file.write(chunk)
            tmp_file_path = tmp_file.name

        # 3. Select Loader and Process
        chunks = loadingDocument(tmp_file_path)

        if not chunks:
            return Response({"error": "No text could be extracted from the document."}, status=status.HTTP_400_BAD_REQUEST)
        
        
        embedder = Embedding(path=uploaded_file.name, chunks=chunks)
        
        # Configure Model (You can make this dynamic via request.data later)
        
        embedder.set_embedding_model(
            model_type='ollama', 
            model_name='nomic-embed-text:latest'
        )

        result = embedder.create_or_load_vectorstore(userId="guest")

        retriever = result["retriever"]
        collection_name = result["collection_name"]
        persist_path = result["persist_path"]
        



        # 4. Success Response
        return Response({
            "status": "success",
            "file_name": uploaded_file.name,
            "persisted_at": persist_path,
            "collection_name": collection_name,
            "chunks_count": len(chunks),
            "file_type": file_type,
            # Returning the first 100 chars of the first chunk as a preview
            "preview": chunks[0].page_content[:100] if chunks else "No text extracted"
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    finally:
        # 5. Cleanup: Delete the temp file from disk
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)