from typing import AsyncGenerator, Optional
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db, AsyncSessionLocal
from app.storage.minio_client import MinioClient, get_minio_client
from app.ml.detector import YOLOv8Agnostic
from app.ml.encoder import DINOv2Encoder
from app.services.object_service import ObjectService
from app.services.session_service import SessionService
from app.services.counting_service import CountingService
from app.core.config import settings


# ==========================================
# ML Model Singletons
# ==========================================

_yolo_model: Optional[YOLOv8Agnostic] = None
_dino_encoder: Optional[DINOv2Encoder] = None


def get_yolo_model() -> YOLOv8Agnostic:
    """Get YOLO model singleton."""
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLOv8Agnostic(
            model_path=settings.YOLO_MODEL_PATH,
            conf_threshold=settings.DETECTION_CONFIDENCE
        )
    return _yolo_model


def get_dino_encoder() -> DINOv2Encoder:
    """Get DINOv2 encoder singleton."""
    global _dino_encoder
    if _dino_encoder is None:
        _dino_encoder = DINOv2Encoder(
            model_size=settings.DINO_MODEL_SIZE,
            model_path=settings.DINO_MODEL_PATH
        )
    return _dino_encoder


def init_ml_models() -> None:
    """Initialize ML models at startup."""
    get_yolo_model()
    get_dino_encoder()


def cleanup_ml_models() -> None:
    """Cleanup ML models."""
    global _yolo_model, _dino_encoder
    _yolo_model = None
    _dino_encoder = None


# ==========================================
# Service Dependencies
# ==========================================

async def get_object_service(
    db: AsyncSession = Depends(get_db),
    dino: DINOv2Encoder = Depends(get_dino_encoder),
    minio: MinioClient = Depends(get_minio_client)
) -> ObjectService:
    """Get ObjectService instance."""
    return ObjectService(db=db, dino_encoder=dino, minio_client=minio)


# Session service singleton for maintaining in-memory state
_session_service: Optional[SessionService] = None


async def get_session_service(
    db: AsyncSession = Depends(get_db)
) -> SessionService:
    """Get SessionService instance."""
    # Note: We create a new service per request but the in-memory state
    # is managed through a shared mechanism
    return SessionService(db=db)


# Counting service singleton
_counting_service: Optional[CountingService] = None


async def get_counting_service(
    yolo: YOLOv8Agnostic = Depends(get_yolo_model),
    dino: DINOv2Encoder = Depends(get_dino_encoder),
    session_service: SessionService = Depends(get_session_service)
) -> CountingService:
    """Get CountingService instance."""
    global _counting_service
    if _counting_service is None:
        _counting_service = CountingService(
            yolo=yolo,
            dino=dino,
            session_service=session_service
        )
    return _counting_service
