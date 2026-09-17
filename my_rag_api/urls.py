# my_rag_api/urls.py
from django.urls import path
from .views import *

urlpatterns = [
    path('upload/', upload_document, name='ingest_document'),
    path('query/', query_document, name='query_document'),
]