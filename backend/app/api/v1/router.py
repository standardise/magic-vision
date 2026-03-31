from fastapi import APIRouter

from app.api.v1.endpoints.objects import router as objects_router
from app.api.v1.endpoints.sessions import router as sessions_router
from app.api.v1.endpoints.stream import router as stream_router


api_router = APIRouter(prefix="/api/v1")

api_router.include_router(objects_router)
api_router.include_router(sessions_router)
api_router.include_router(stream_router)
