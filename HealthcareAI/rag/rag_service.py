import os
import json
import threading
from typing import List, Dict, Any
from pathlib import Path

from config.config import settings
from utils.logger import logger
from rag.document_loader import DocumentLoader
from rag.text_splitter import ProtocolTextSplitter
from rag.vector_store import LocalVectorStore

class RAGService:
    """Coordinates document parsing, text chunking, indexing, retrieval, and auto-rebuild checking."""
    
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RAGService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.protocols_dir = settings.get_absolute_path(settings.PROTOCOLS_DIR)
        self.vector_store_dir = settings.get_absolute_path(settings.VECTOR_STORE_DIR)
        self.metadata_path = os.path.join(self.vector_store_dir, "index_metadata.json")
        
        # Ensure directories exist
        os.makedirs(self.protocols_dir, exist_ok=True)
        os.makedirs(self.vector_store_dir, exist_ok=True)
        
        self.vector_store = LocalVectorStore(self.vector_store_dir)
        self.text_splitter = ProtocolTextSplitter()
        self._rebuild_lock = threading.Lock()
        
        # Load vector store index on initialization if it exists
        self.vector_store.load_index()
        self._initialized = True
        logger.info("RAGService initialized.")

    def _get_current_directory_state(self) -> Dict[str, float]:
        """Scans the protocols directory and returns a map of filenames to their modification times."""
        state = {}
        if not os.path.isdir(self.protocols_dir):
            return state
            
        for root, _, files in os.walk(self.protocols_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in [".pdf", ".txt"]:
                    file_path = os.path.join(root, file)
                    try:
                        state[file] = os.path.getmtime(file_path)
                    except Exception as e:
                        logger.warning(f"Could not read modification time for {file}: {e}")
        return state

    def _load_metadata(self) -> Dict[str, float]:
        """Loads index metadata containing files state from disk."""
        if not os.path.exists(self.metadata_path):
            return {}
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading index metadata: {e}")
            return {}

    def _write_metadata(self, state: Dict[str, float]):
        """Writes current files state to index metadata JSON."""
        try:
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving index metadata: {e}")

    def check_and_rebuild_index(self) -> bool:
        """
        Scans protocols directory. Rebuilds FAISS index if:
        1. No FAISS index currently exists on disk.
        2. Any file is added, modified, or deleted from protocols directory.
        Returns True if rebuild occurred, False otherwise.
        """
        with self._rebuild_lock:
            current_state = self._get_current_directory_state()
            metadata = self._load_metadata()
            faiss_exists = os.path.exists(os.path.join(self.vector_store_dir, "index.faiss"))
            
            # Rebuild is required if index doesn't exist, or file states changed
            needs_rebuild = not faiss_exists or current_state != metadata
            
            if needs_rebuild:
                logger.info("Changes in medical protocols detected. Triggering automatic index rebuild...")
                # 1. Load documents
                documents = DocumentLoader.load_directory(self.protocols_dir)
                if not documents:
                    logger.warning("No clinical protocols found. Vector store will remain unbuilt or empty.")
                    # Clean up old FAISS if existing files were deleted
                    if faiss_exists:
                        try:
                            os.remove(os.path.join(self.vector_store_dir, "index.faiss"))
                            os.remove(os.path.join(self.vector_store_dir, "index.pkl"))
                            if os.path.exists(self.metadata_path):
                                os.remove(self.metadata_path)
                            self.vector_store.db = None
                            logger.info("Cleared old index files as protocols directory is empty.")
                        except Exception as e:
                            logger.error(f"Failed to clear old index: {e}")
                    return True
                
                # 2. Chunk documents
                chunks = self.text_splitter.split_documents(documents)
                
                # 3. Build & save
                success = self.vector_store.build_and_save_index(chunks)
                if success:
                    # 4. Save metadata state
                    self._write_metadata(current_state)
                    logger.info("Automatic index rebuild completed successfully.")
                    return True
                else:
                    logger.error("Failed to automatically rebuild the vector index.")
                    return False
            
            logger.debug("No protocol changes detected. Skipping index rebuild.")
            return False

    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves matching document chunks for a query.
        Automatically checks and rebuilds index if directory states changed.
        """
        # Always run check/rebuild before query to guarantee data freshness
        self.check_and_rebuild_index()
        
        results = self.vector_store.similarity_search(query, top_k=top_k)
        
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        return formatted_results
