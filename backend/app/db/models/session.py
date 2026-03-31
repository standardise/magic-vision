from sqlalchemy import Column, String, Text, Integer, DateTime, JSON, ARRAY
from sqlalchemy.sql import func
import uuid

from app.db.session import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class CountingSession(Base):
    """Counting session - tracks counting history."""
    __tablename__ = "counting_sessions"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Target objects to count (array of object IDs)
    target_object_ids = Column(ARRAY(String(36)), nullable=False, default=[])
    
    # Counting results
    class_counts = Column(JSON, default={})  # {"object_name": count}
    total_count = Column(Integer, default=0)
    
    # Session state
    status = Column(String(50), default="created", index=True)  # created, running, stopped, cancelled
    camera_source = Column(String(255), nullable=True)  # webcam, rtsp://..., usb:0
    
    # Timestamps
    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<CountingSession(id={self.id}, name={self.name}, status={self.status})>"
