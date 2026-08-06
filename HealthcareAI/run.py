import os
import sys
import uvicorn
from config.config import settings
from utils.logger import logger
from rag.rag_service import RAGService

BANNER = """
========================================================================
    ___   _____ _   _   ___         ___  ___ 
   / _ \ /  ___| | | | / _ \        |  \/  | 
  / /_\ \\ `--.| |_| |/ /_\ \       | .  . | ___  ___ 
  |  _  | `--. \  _  ||  _  |       | |\/| |/ _ \/ __|
  | | | |/\__/ / | | || | | |       | |  | |  __/ (__ 
  \_| |_/\____/\_| |_/\_| |_/       \_|  |_/\___|\___|
                                                     
      ASHA/ANM Offline AI Clinical Assistant Server
      Powered by SentenceTransformers, FAISS, & Phi-3
========================================================================
"""

def prepare_system():
    """Validates local environment paths and syncs active medical guidelines on boot."""
    print(BANNER)
    logger.info("Initializing system boot procedures...")

    # Create workspace path requirements if missing
    directories = ["logs", "data/protocols", "vector_store", "database"]
    for directory in directories:
        dir_path = settings.get_absolute_path(directory)
        if not os.path.exists(dir_path):
            logger.info(f"Creating required directory footprint: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)

    # Sync and build vector store database
    try:
        logger.info("Syncing and indexing offline medical protocols...")
        rag = RAGService()
        rag.check_and_rebuild_index()
        logger.info("Protocols database sync completed successfully.")
    except Exception as e:
        logger.error(f"Failed to sync medical protocols database on startup: {e}")
        print(f"Warning: Medical guidelines index sync failed: {e}")

def main():
    """Boots the uvicorn REST server."""
    prepare_system()
    
    host = settings.HOST
    port = settings.PORT
    log_level = settings.LOG_LEVEL.lower()

    logger.info(f"Launching Uvicorn server on http://{host}:{port} (Log Level: {log_level})...")
    print(f"[*] API Status endpoint active: http://{host}:{port}/api/status")
    print(f"[*] Assistant REST routing: http://{host}:{port}/api/assistant/interact")
    print("Press Ctrl+C to stop the clinical server.")
    print("========================================================================\n")

    try:
        uvicorn.run(
            "backend.app:app",
            host=host,
            port=port,
            log_level=log_level,
            reload=False # Set to false to prevent multiple thread loads in production environments
        )
    except KeyboardInterrupt:
        logger.info("Shutting down clinical assistant server.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Server encounter startup failure: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
