from fastapi import APIRouter, Depends, Query, status
from typing import List, Optional

from app.api.v1.deps import get_session_service
from app.services.session_service import SessionService
from app.schemas import (
    SessionCreateRequest,
    SessionResponse,
    SessionDetailResponse,
    SessionStopResponse
)


router = APIRouter(prefix="/sessions", tags=["Sessions"])


@router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create counting session"
)
async def create_session(
    request: SessionCreateRequest,
    service: SessionService = Depends(get_session_service)
):
    """
    Create a new counting session.
    
    - Specify which objects to count via target_object_ids
    - Optionally specify camera source (webcam, RTSP, USB)
    - Session is created but not started
    """
    return await service.create_session(request)


@router.get(
    "",
    response_model=List[SessionResponse],
    summary="List all sessions"
)
async def list_sessions(
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    service: SessionService = Depends(get_session_service)
):
    """Get list of all counting sessions (history)."""
    return await service.list_sessions(
        status=status_filter,
        limit=limit,
        offset=offset
    )


@router.get(
    "/{session_id}",
    response_model=SessionDetailResponse,
    summary="Get session details"
)
async def get_session(
    session_id: str,
    service: SessionService = Depends(get_session_service)
):
    """Get detailed information about a session including live counts if running."""
    return await service.get_session(session_id)


@router.post(
    "/{session_id}/start",
    response_model=SessionResponse,
    summary="Start counting session"
)
async def start_session(
    session_id: str,
    service: SessionService = Depends(get_session_service)
):
    """
    Start a counting session.
    
    - Initializes counting state
    - Sets status to 'running'
    - Connect to WebSocket endpoint to stream video
    """
    return await service.start_session(session_id)


@router.post(
    "/{session_id}/stop",
    response_model=SessionStopResponse,
    summary="Stop counting session"
)
async def stop_session(
    session_id: str,
    service: SessionService = Depends(get_session_service)
):
    """
    Stop a running session and save final counts.
    
    - Saves class_counts and total_count
    - Sets status to 'stopped'
    """
    return await service.stop_session(session_id)


@router.post(
    "/{session_id}/cancel",
    response_model=SessionResponse,
    summary="Cancel session"
)
async def cancel_session(
    session_id: str,
    service: SessionService = Depends(get_session_service)
):
    """Cancel a session without saving results."""
    return await service.cancel_session(session_id)


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete session"
)
async def delete_session(
    session_id: str,
    service: SessionService = Depends(get_session_service)
):
    """Delete a session from history. Cannot delete running sessions."""
    await service.delete_session(session_id)


@router.get(
    "/{session_id}/counts",
    summary="Get live counts"
)
async def get_live_counts(
    session_id: str,
    service: SessionService = Depends(get_session_service)
):
    """Get current counts for a running session."""
    counts = service.get_live_counts(session_id)
    if counts:
        return counts
    # Fall back to database
    session = await service.get_session(session_id)
    return {
        "class_counts": session.class_counts,
        "total_count": session.total_count
    }
