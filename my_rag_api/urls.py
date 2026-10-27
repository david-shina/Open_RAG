from django.urls import path

from .views import health_check, query_document, upload_document

urlpatterns = [
    path("health/", health_check, name="health_check"),
    path("upload/", upload_document, name="upload_document"),
    path("query/", query_document, name="query_document"),
]
