import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from config.config import settings
from utils.logger import logger
from backend.routes import assistant_router, status_router, protocols_router, history_router
from rag.rag_service import RAGService

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Check and rebuild vector store
    logger.info("Initializing RAG service on startup...")
    try:
        rag_service = RAGService()
        rebuilt = rag_service.check_and_rebuild_index()
        if rebuilt:
            logger.info("Vector database rebuild check complete: rebuilt.")
        else:
            logger.info("Vector database rebuild check complete: up-to-date.")
    except Exception as e:
        logger.error(f"Error running startup index rebuild: {e}")
    yield
    # Shutdown
    logger.info("Shutting down API server...")

# Initialize FastAPI App with metadata for docs
app = FastAPI(
    title="Offline AI Clinical Assistant API",
    description="ASHA/ANM healthcare worker AI clinical assistant offline server.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration for Flutter/web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for request/response logging & timing
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    path = request.url.path
    method = request.method
    logger.info(f"Incoming request: {method} {path}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Completed request: {method} {path} - Status: {response.status_code} - Duration: {process_time:.4f}s")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Failed request: {method} {path} - Error: {str(e)} - Duration: {process_time:.4f}s")
        raise e

# Custom Global Exception Handler for HTTP exceptions
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTPException on {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "detail": exc.detail}
    )

# General Exception Handler
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "detail": "An internal server error occurred."}
    )

# Include Routers
app.include_router(status_router, prefix="/api")
app.include_router(assistant_router, prefix="/api")
app.include_router(protocols_router, prefix="/api")
app.include_router(history_router, prefix="/api")

# Static mounting for generated Audio files (TTS outputs)
audio_dir = settings.get_absolute_path(settings.TTS_AUDIO_OUTPUT_DIR)
logger.info(f"Mounting static files from: {audio_dir}")
app.mount("/audio", StaticFiles(directory=audio_dir), name="audio")

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Offline AI Clinical Assistant Backend API.",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server at {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "backend.app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
