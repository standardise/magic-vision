from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # App
    APP_NAME: str = "Magic Vision API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # PostgreSQL
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_USER: str = Field(default="magicvision")
    POSTGRES_PASSWORD: str = Field(default="!magicvision1")
    POSTGRES_DB: str = Field(default="magicvision")
    
    @property
    def database_url(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @property
    def database_url_sync(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # MinIO
    MINIO_HOST: str = Field(default="localhost")
    MINIO_PORT: int = Field(default=9000)
    MINIO_ROOT_USER: str = Field(default="minioadmin")
    MINIO_ROOT_PASSWORD: str = Field(default="minioadmin")
    MINIO_BUCKET: str = Field(default="magic-vision")
    MINIO_SECURE: bool = Field(default=False)
    
    @property
    def minio_endpoint(self) -> str:
        return f"{self.MINIO_HOST}:{self.MINIO_PORT}"
    
    # ML Models - paths relative to backend folder
    YOLO_MODEL_PATH: str = Field(default="models/yolov8n.pt")
    DINO_MODEL_SIZE: str = Field(default="vits14")
    DINO_MODEL_PATH: Optional[str] = Field(default="models/dinov2_vits14.pth")
    EMBEDDING_DIM: int = Field(default=384)  # DINOv2 ViT-S/14
    
    # Detection settings
    DETECTION_CONFIDENCE: float = Field(default=0.25)
    SIMILARITY_THRESHOLD: float = Field(default=0.7)
    VOTE_COUNT: int = Field(default=3)  # Votes needed to confirm class
    
    # Confidence thresholds for matching
    MIN_MATCH_CONFIDENCE: float = Field(default=0.65)  # Minimum to accept match
    HIGH_CONFIDENCE_THRESHOLD: float = Field(default=0.85)  # Skip voting if above this
    
    # Stream settings
    FRAME_SKIP: int = Field(default=5)  # Process every N frames
    MAX_TRACK_AGE: int = Field(default=30)  # Frames before track is lost
    TRACK_MEMORY_MAX_AGE: int = Field(default=150)  # Frames before track memory cleanup
    
    # Batch processing
    BATCH_ENCODING: bool = Field(default=True)  # Enable batch DINOv2 encoding
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
