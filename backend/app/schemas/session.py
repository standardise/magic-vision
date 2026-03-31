from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime


class SessionCreateRequest(BaseModel):
    """Request to create a new counting session."""
    name: str = Field(..., description="Session name")
    description: Optional[str] = Field(None, description="Session description")
    target_object_ids: List[str] = Field(..., description="List of object IDs to count")
    camera_source: Optional[str] = Field(None, description="Camera source (webcam, rtsp://..., usb:0)")


class SessionStartRequest(BaseModel):
    """Request to start an existing session."""
    session_id: str


class SessionResponse(BaseModel):
    """Session response."""
    id: str
    name: str
    description: Optional[str] = None
    target_object_ids: List[str] = []
    target_object_names: Dict[str, str] = {}  # {object_id: object_name}
    class_counts: Dict[str, int] = {}
    total_count: int = 0
    status: str = "created"
    camera_source: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class SessionDetailResponse(SessionResponse):
    """Detailed session response."""
    pass


class SessionStopResponse(BaseModel):
    """Response when stopping a session."""
    id: str
    name: str
    status: str
    class_counts: Dict[str, int]
    total_count: int
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class SessionListResponse(BaseModel):
    """Paginated list of sessions."""
    sessions: List[SessionResponse]
    total: int
    page: int
    per_page: int