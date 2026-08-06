import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from utils.logger import logger

class DocumentLoader:
    """Document loader that scans a directory and extracts content from PDFs and TXT files."""
    
    @staticmethod
    def load_file(file_path: str) -> List[Document]:
        """Loads a single file (PDF or TXT) and returns a list of Documents."""
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return []
            
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".pdf":
                logger.info(f"Loading PDF file: {file_path}")
                loader = PyPDFLoader(file_path)
                return loader.load()
            elif ext == ".txt":
                logger.info(f"Loading TXT file: {file_path}")
                loader = TextLoader(file_path, encoding="utf-8")
                return loader.load()
            else:
                logger.warning(f"Unsupported file type: {ext} for file {file_path}")
                return []
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}", exc_info=True)
            return []

    @classmethod
    def load_directory(cls, directory_path: str) -> List[Document]:
        """Scans directory_path for PDF and TXT files and returns aggregated Documents."""
        if not os.path.isdir(directory_path):
            logger.warning(f"Directory not found: {directory_path}")
            return []
            
        documents = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                docs = cls.load_file(file_path)
                documents.extend(docs)
                
        logger.info(f"Loaded {len(documents)} page/document chunks from {directory_path}")
        return documents
