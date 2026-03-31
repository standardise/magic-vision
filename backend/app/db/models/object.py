from sqlalchemy import Column, String, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
import uuid

from app.db.session import Base
from app.core.config import settings


def generate_uuid() -> str:
    return str(uuid.uuid4())


class Object(Base):
    """Object library - stores reference objects with embeddings."""
    __tablename__ = "objects"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    embedding = Column(Vector(settings.EMBEDDING_DIM), nullable=True)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    images = relationship("ObjectImage", back_populates="object", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Object(id={self.id}, name={self.name})>"


class ObjectImage(Base):
    """Reference images for objects stored in MinIO."""
    __tablename__ = "object_images"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    object_id = Column(String(36), ForeignKey("objects.id", ondelete="CASCADE"), nullable=False, index=True)
    image_path = Column(String(512), nullable=False)  # MinIO path
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    object = relationship("Object", back_populates="images")
    
    def __repr__(self):
        return f"<ObjectImage(id={self.id}, object_id={self.object_id})>"
