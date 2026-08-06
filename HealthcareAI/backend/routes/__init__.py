from backend.routes.assistant import router as assistant_router
from backend.routes.status import router as status_router
from backend.routes.protocols import router as protocols_router
from backend.routes.history import router as history_router

__all__ = [
    "assistant_router",
    "status_router",
    "protocols_router",
    "history_router",
]
