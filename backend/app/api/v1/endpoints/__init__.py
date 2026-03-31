from app.api.v1.endpoints.objects import router as objects_router
from app.api.v1.endpoints.sessions import router as sessions_router
from app.api.v1.endpoints.stream import router as stream_router

__all__ = ["objects_router", "sessions_router", "stream_router"]
