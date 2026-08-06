from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger import logger

class ProtocolTextSplitter:
    """Recursively splits clinical protocol documents into optimal chunks for offline vector indexing."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        logger.info(f"Initialized ProtocolTextSplitter with chunk_size={chunk_size}, overlap={chunk_overlap}")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits a list of raw documents into smaller chunked documents."""
        if not documents:
            return []
        chunks = self.splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} original documents into {len(chunks)} text chunks.")
        return chunks
