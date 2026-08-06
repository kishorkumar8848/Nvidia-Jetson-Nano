from fastapi import APIRouter
from config.config import settings
from utils.logger import logger

router = APIRouter(prefix="/status", tags=["System Status"])

@router.get("")
async def get_status():
    """Returns the system health and model configuration details."""
    logger.info("Status endpoint requested.")
    
    # We could query device specs or Ollama health here, currently returning a mockup of health indicators
    return {
        "status": "healthy",
        "device": "NVIDIA Jetson Orin Nano",
        "configurations": {
            "llm_model": settings.LLM_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL_NAME,
            "asr_provider": settings.ASR_MODEL_PROVIDER,
            "translation_provider": "IndicTrans2",
            "vision_provider": settings.VISION_MODEL_PROVIDER,
            "tts_provider": settings.TTS_MODEL_PROVIDER
        }
    }
