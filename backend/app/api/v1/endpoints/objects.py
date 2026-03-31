from fastapi import APIRouter, Depends, UploadFile, File, Form, Query, status
from typing import List, Optional, Annotated

from app.api.v1.deps import get_object_service
from app.services.object_service import ObjectService
from app.schemas import (
    ObjectResponse, 
    ObjectDetailResponse, 
    ObjectUpdateRequest
)


router = APIRouter(prefix="/objects", tags=["Objects"])


@router.post(
    "",
    response_model=ObjectResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create new object",
    description="Create a new object with 3-10 reference images for few-shot recognition."
)
async def create_object(
    name: Annotated[str, Form(description="Object name")],
    files: Annotated[List[UploadFile], File(description="3-10 reference images (JPEG/PNG)")],
    description: Annotated[Optional[str], Form(description="Object description")] = None,
    service: ObjectService = Depends(get_object_service)
):
    """
    Create a new object in the library.
    
    - Upload 3-10 reference images
    - System encodes images with DINOv2
    - Creates prototype embedding for few-shot matching
    """
    return await service.create_object(
        name=name,
        description=description,
        files=files
    )


@router.get(
    "",
    response_model=List[ObjectResponse],
    summary="List all objects"
)
async def list_objects(
    active_only: bool = Query(True, description="Show only active objects"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    service: ObjectService = Depends(get_object_service)
):
    """Get list of all objects in the library."""
    return await service.list_objects(
        active_only=active_only,
        limit=limit,
        offset=offset
    )


@router.get(
    "/{object_id}",
    response_model=ObjectDetailResponse,
    summary="Get object details"
)
async def get_object(
    object_id: str,
    service: ObjectService = Depends(get_object_service)
):
    """Get detailed information about a specific object."""
    return await service.get_object(object_id)


@router.put(
    "/{object_id}",
    response_model=ObjectResponse,
    summary="Update object metadata"
)
async def update_object(
    object_id: str,
    request: ObjectUpdateRequest,
    service: ObjectService = Depends(get_object_service)
):
    """Update object name or description."""
    return await service.update_object(
        object_id=object_id,
        name=request.name,
        description=request.description
    )


@router.post(
    "/{object_id}/images",
    response_model=ObjectResponse,
    summary="Add more reference images"
)
async def add_images(
    object_id: str,
    files: Annotated[List[UploadFile], File(description="Additional reference images (JPEG/PNG)")],
    service: ObjectService = Depends(get_object_service)
):
    """
    Add more reference images to an existing object.
    
    - Maximum 10 images total per object
    - System recalculates prototype embedding
    """
    return await service.add_images(object_id=object_id, files=files)


@router.delete(
    "/{object_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete object"
)
async def delete_object(
    object_id: str,
    hard: bool = Query(False, description="Permanently delete (including images)"),
    service: ObjectService = Depends(get_object_service)
):
    """
    Delete an object from the library.
    
    - Soft delete (default): Object is deactivated but data remains
    - Hard delete: Object and images permanently removed
    """
    await service.delete_object(object_id=object_id, hard=hard)


@router.post(
    "/{object_id}/reactivate",
    response_model=ObjectResponse,
    summary="Reactivate soft-deleted object"
)
async def reactivate_object(
    object_id: str,
    service: ObjectService = Depends(get_object_service)
):
    """Reactivate a previously soft-deleted object."""
    return await service.reactivate_object(object_id)
