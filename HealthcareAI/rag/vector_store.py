import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rag.embeddings import get_embedding_model
from utils.logger import logger

class LocalVectorStore:
    """Manages the local FAISS vector database life cycle (load, save, rebuild, search)."""
    
    def __init__(self, store_dir: str):
        self.store_dir = store_dir
        self.embeddings = get_embedding_model()
        self.db: Optional[FAISS] = None
        logger.info(f"Initialized LocalVectorStore targeting: {store_dir}")

    def load_index(self) -> bool:
        """Loads the FAISS index from disk. Returns True if successful, False otherwise."""
        if not os.path.exists(os.path.join(self.store_dir, "index.faiss")):
            logger.warning(f"No FAISS index found at {self.store_dir}")
            return False
        try:
            logger.info(f"Loading local FAISS index from {self.store_dir}...")
            self.db = FAISS.load_local(
                folder_path=self.store_dir,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("FAISS index loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Error loading local FAISS index: {e}", exc_info=True)
            return False

    def build_and_save_index(self, chunks: List[Document]) -> bool:
        """Builds a new FAISS index from the provided document chunks and saves it to disk."""
        if not chunks:
            logger.warning("No chunks provided to build vector index.")
            return False
        try:
            logger.info(f"Building FAISS vector index with {len(chunks)} chunks...")
            self.db = FAISS.from_documents(documents=chunks, embedding=self.embeddings)
            
            # Ensure directory exists
            os.makedirs(self.store_dir, exist_ok=True)
            self.db.save_local(folder_path=self.store_dir)
            logger.info(f"FAISS index successfully saved to {self.store_dir}")
            return True
        except Exception as e:
            logger.error(f"Error building or saving FAISS index: {e}", exc_info=True)
            return False

    def similarity_search(self, query: str, top_k: int = 3) -> List[Document]:
        """Performs semantic similarity search for the given query."""
        if self.db is None:
            # Try to load if not already loaded
            if not self.load_index():
                logger.error("Vector store database is not loaded and could not be resolved.")
                return []
        try:
            logger.debug(f"Performing vector similarity search for query: '{query}' (top_k={top_k})")
            results = self.db.similarity_search(query, k=top_k)
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {e}", exc_info=True)
            return []
