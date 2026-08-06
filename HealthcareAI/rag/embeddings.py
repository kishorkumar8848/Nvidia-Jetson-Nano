from langchain_huggingface import HuggingFaceEmbeddings
from config.config import settings
from utils.logger import logger

_embeddings_cache = None

def get_embedding_model() -> HuggingFaceEmbeddings:
    """Retrieves or loads the cached local Sentence Transformers embedding model."""
    global _embeddings_cache
    if _embeddings_cache is None:
        model_name = settings.EMBEDDING_MODEL_NAME
        logger.info(f"Loading Sentence Transformers model: {model_name}...")
        try:
            # HuggingFaceEmbeddings runs completely locally once cached
            _embeddings_cache = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},  # Can switch to "cuda" on Jetson if torch has CUDA active
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info(f"Successfully loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise e
    return _embeddings_cache
