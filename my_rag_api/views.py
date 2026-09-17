import logging
import os
from typing import List, Literal, Dict, Any
from rest_framework.decorators import api_view, parser_classes 
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from rag_demo.queryTranslator import QueryTranslator
from rag_demo.embeddings import Embedding
from rag_demo.prompts import *
from rag_demo.loaders import (pdf_loader, docxloader, pptx_loader)
from pathlib import Path
from .utilis import loadingDocument
from rag_demo.queryTranslator import *
from rag_demo.prompts import *
from langchain_groq import ChatGroq
from django.views.decorators.csrf import csrf_exempt
from drf_spectacular.utils import extend_schema, OpenApiResponse
from drf_spectacular.types import OpenApiTypes


from .serializers import (
    DocumentUploadSerializer,
    QuerySerializer,
    DocumentResponseSerializer,
    QueryResponseSerializer,
    ErrorResponseSerializer

)





logger = logging.getLogger(__name__)

_vector_store_cache = {}


UPLOAD_DIR = Path('media/documents')
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)



@extend_schema(
    summary="Manual File Upload",
    description="Force swagger to show file upload button",
    # We manually define the request body here to force 'binary' mode
    request={
        'multipart/form-data': {
            'type': 'object',
            'properties': {
                'file': {
                    'type': 'string',
                    'format': 'binary'  # <--- This triggers the file picker
                },
                'user_id': {
                    'type': 'string'
                }
            },
            'required': ['file', 'user_id']
        }
    },
    responses={201: dict},
)


@csrf_exempt
@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def upload_document(request):
    try:
        # Validate incoming data
        serializer = DocumentUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {'success': False, 'error': serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Extract validated data
        uploaded_file = serializer.validated_data['file']
        user_id = serializer.validated_data['user_id']
        
        # Create user-specific directory
        user_dir = UPLOAD_DIR / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file with unique name (use original filename)
        file_name = uploaded_file.name
        file_path = user_dir / file_name
        
        # Save the uploaded file
        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        
        logger.info(f"File saved: {file_path}")
        
        # Process document: load and chunk
        try:
            chunks = loadingDocument(str(file_path))
            logger.info(f"Document chunked: {len(chunks)} chunks created")
        except Exception as e:
            logger.error(f"Error loading document: {str(e)}")
            # Clean up the file if processing fails
            file_path.unlink(missing_ok=True)
            return Response(
                {
                    'success': False,
                    'error': 'Failed to process document',
                    'details': str(e)
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Create embeddings and vector store
        try:
           
            embedder = Embedding(path=str(file_path), chunks=chunks)
            embedder.set_embedding_model(
            model_type='ollama', 
            model_name='nomic-embed-text:latest'
            )

            result = embedder.create_or_load_vectorstore(userId=str(user_id))

            retriever = result["retriever"]
            collection_name = result["collection_name"]
            persist_path = result["persist_path"]

            cache_key = f"{user_id}:{file_name}"
            print("CACHE KEY:", cache_key)
            _vector_store_cache[cache_key] = {
                'retriever': retriever,
                'file_path': str(file_path),
                'chunks_count': len(chunks),
                
            }

            logger.info(f"Vector store created and cached: {cache_key}")
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            file_path.unlink(missing_ok=True)
            return Response(
                {
                    'success': False,
                    'error': 'Failed to create embeddings',
                    'details': str(e)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        # Return success response
        response_data = {
            'success': True,
            'message': 'Document uploaded and processed successfully',
            'document_id': file_name,
            'document_path': str(file_path),
            'chunks_created': len(chunks),
            'persist_path': persist_path,
            'user_id': user_id,
            # 'vector_store_cache': str(_vector_store_cache[cache_key])
        }
        
        return Response(response_data, status=status.HTTP_201_CREATED)
    
    except Exception as e:
        logger.error(f"Unexpected error in upload_document: {str(e)}")
        return Response(
            {
                'success': False,
                'error': 'An unexpected error occurred',
                'details': str(e)
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )



@extend_schema(
    summary="Query a Document",
    description="Generates RAG answers based on uploaded documents.",
    request=QuerySerializer,
    responses={200: dict, 404: dict}
)
@api_view(['POST'])
def query_document(request):

    try:
        # Validate incoming data
        serializer = QuerySerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {'success': False, 'error': serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        user_id = serializer.validated_data['user_id']
        document_id = serializer.validated_data['document_id']
        question = serializer.validated_data['question']
        cache_key = f"{user_id}:{document_id}"
        print("CACHE KEY:", cache_key)
        if cache_key not in _vector_store_cache:
            return Response(
                {
                    'success': False,
                    'error': 'Document not found or not processed for this user.'
                },
                status=status.HTTP_404_NOT_FOUND
            )
        retriever = _vector_store_cache[cache_key]['retriever']


        translator = QueryTranslator(n_variants=3)
        generated_queries = translator.translate(question, mode='hybrid')

        unique_docs = {}
        for q in generated_queries:
            docs = retriever.invoke(q)
            for doc in docs:
                unique_docs[doc.page_content] = doc

        retrieved_docs_list = list(unique_docs.values())
        print(str(retrieved_docs_list))

        if not retrieved_docs_list:
            return Response({
                'success': True,
                'answer': "I couldn't find any relevant information in the document to answer your question.",
                'sources': []
            })
    
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs_list])
        try:
            # Initialize LLM (Ensure GROQ_API_KEY is in .env)
            llm = ChatGroq(
                model="openai/gpt-oss-120b",  # Or 'mixtral-8x7b-32768' for better reasoning
                temperature=0
                )

            # Create the Chain
            rag_chain = rag_prompt | llm | StrOutputParser()

            # Run the Chain
            final_answer = rag_chain.invoke({
                "context": context_text,
                "question": question
                })

        except Exception as llm_error:
            logger.error(f"LLM Generation failed: {llm_error}")
            return Response({'success': False, 'error': 'Failed to generate answer from LLM'}, status=500)

        # 7. Return Final Response
        return Response({
            'success': True,
            'original_query': question,
                'answer': final_answer,
                'generated_queries': generated_queries,
                'source_chunks': len(retrieved_docs_list)
            }, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return Response({'success': False, 'error': str(e)}, status=500)







    
    