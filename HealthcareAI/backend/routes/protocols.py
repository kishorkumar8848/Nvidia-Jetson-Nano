import os
from fastapi import APIRouter, File, UploadFile, HTTPException
from config.config import settings
from utils.logger import logger
from rag.rag_service import RAGService

router = APIRouter(prefix="/protocols", tags=["RAG Protocol Management"])

@router.post("/upload")
async def upload_protocol(file: UploadFile = File(...)):
    """Uploads a clinical protocol document (PDF/TXT) and automatically triggers RAG rebuild."""
    logger.info(f"Uploading file: {file.filename}")
    
    # Restrict file types
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdf", ".txt"]:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are allowed.")
    
    protocols_dir = settings.get_absolute_path(settings.PROTOCOLS_DIR)
    os.makedirs(protocols_dir, exist_ok=True)
    
    file_path = os.path.join(protocols_dir, file.filename)
    
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        logger.info(f"File saved successfully to {file_path}")
        
        # Trigger RAG check and rebuild
        rag_service = RAGService()
        rebuilt = rag_service.check_and_rebuild_index()
        indexed_count = len(rag_service._load_metadata())
        
        return {
            "success": True, 
            "filename": file.filename, 
            "saved_path": file_path, 
            "rebuilt_occurred": rebuilt,
            "total_indexed_protocols": indexed_count
        }
    except Exception as e:
        logger.error(f"Failed to save protocol file: {e}")
        raise HTTPException(status_code=500, detail=f"File save/index error: {str(e)}")

@router.post("/rebuild")
async def rebuild_vector_store():
    """Manually triggers the pipeline to rebuild the RAG FAISS vector store index."""
    logger.info("Manual trigger for vector store rebuild.")
    try:
        rag_service = RAGService()
        rebuilt = rag_service.check_and_rebuild_index()
        indexed_count = len(rag_service._load_metadata())
        return {
            "success": True,
            "message": "Vector store rebuild executed successfully.",
            "rebuilt_occurred": rebuilt,
            "indexed_protocols_count": indexed_count
        }
    except Exception as e:
        logger.error(f"Failed manual rebuild of vector store: {e}")
        raise HTTPException(status_code=500, detail=f"Rebuild error: {str(e)}")

