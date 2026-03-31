from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


class ObjectBase(BaseModel):
    name: str = Field(..., description="Object name (e.g., 'Screw_M4')")
    description: Optional[str] = Field(None, description="Object description")


class ObjectCreateRequest(ObjectBase):
    """Request for creating a new object (images sent as multipart form)."""
    pass


class ObjectUpdateRequest(BaseModel):
    """Request for updating object metadata."""
    name: Optional[str] = None
    description: Optional[str] = None


class ObjectResponse(BaseModel):
    """Object response for list views."""
    id: str
    name: str
    description: Optional[str] = None
    image_urls: List[str] = []
    image_count: int = 0
    is_active: bool = True
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class ObjectDetailResponse(ObjectResponse):
    """Detailed object response including embedding status."""
    has_embedding: bool = False


class ObjectListResponse(BaseModel):
    """Paginated list of objects."""
    objects: List[ObjectResponse]
    total: int
    page: int
    per_page: int