import os
from abc import abstractmethod, ABC
from langchain_core.documents import Document

class BaseLoader(ABC):
    """
    Abstract base class for all document loaders (PDF, DOCX, PPTX, etc.).
    Each loader must implement `detect_type` and `load`.
    """

    def __init__(self, file_path:str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File does not exists: {file_path}')
        self.file_path = file_path

    @abstractmethod
    def detect_type(self)->str:
        """Detect document type (text-based, image-based, etc.)."""
        pass

    @abstractmethod
    def load(self) -> list[Document]:
        """Extract and return documents as a list of LangChain Document objects."""
        pass