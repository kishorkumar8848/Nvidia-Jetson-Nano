import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base Directory path
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # General Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Database Settings
    SQLITE_DB_PATH: str = "database/assistant.db"

    # LLM & Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "phi3:mini"

    # RAG Configuration
    PROTOCOLS_DIR: str = "data/protocols"
    VECTOR_STORE_DIR: str = "vector_store"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ASR Configuration
    ASR_MODEL_PROVIDER: str = "bhashini"
    INDIC_CONFORMER_MODEL_PATH: str = "models/indicconformer"

    # Translation Configuration
    INDICTRANS2_MODEL_DIR: str = "models/indictrans2"

    # Vision Configuration
    VISION_MODEL_PROVIDER: str = "moondream"
    VISION_MODEL_PATH: str = "models/moondream"

    # TTS Configuration
    TTS_MODEL_PROVIDER: str = "offline"
    TTS_AUDIO_OUTPUT_DIR: str = "logs"

    # Pydantic Configuration
    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def get_absolute_path(self, relative_path: str) -> str:
        """Helper to get an absolute path resolved against the project base directory."""
        path = Path(relative_path)
        if path.is_absolute():
            return str(path)
        return str((BASE_DIR / path).resolve())

# Instantiate settings
settings = Settings()
