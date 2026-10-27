import os

import pytest
from langchain_core.documents import Document

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "open_rag_final.settings")
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "false"


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="The Eiffel Tower is in Paris and was completed in 1889.",
            metadata={"source": "paris.pdf", "page": 1, "chunk_index": 0},
        ),
        Document(
            page_content="The Great Pyramid of Giza is the oldest ancient wonder.",
            metadata={"source": "egypt.pdf", "page": 1, "chunk_index": 0},
        ),
    ]
