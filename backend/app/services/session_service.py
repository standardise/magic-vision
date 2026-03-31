import numpy as np
from typing import Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
import logging

from app.repositories.session_repository import SessionRepository
from app.repositories.object_repository import ObjectRepository
from app.core.exceptions import (
    SessionNotFoundError, 
    SessionAlreadyRunningError, 
    SessionNotRunningError,
    NoTargetObjectsError,
    ObjectNotFoundError
)
from app.schemas.session import (
    SessionResponse, 
    SessionDetailResponse,
    SessionCreateRequest,
    SessionStopResponse
)

logger = logging.getLogger(__name__)


class SessionService:
    """Business logic for counting session management."""
    
    # Class-level shared state for active sessions across all instances
    # {session_id: {"class_counts": {}, "total_count": 0, "counted_ids": set()}}
    _active_sessions: Dict[str, Dict] = {}
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.session_repo = SessionRepository(db)
        self.object_repo = ObjectRepository(db)
    
    async def create_session(
        self,
        request: SessionCreateRequest
    ) -> SessionResponse:
        """Create a new counting session."""
        if not request.target_object_ids:
            raise NoTargetObjectsError()
        
        # Verify all target objects exist and are active
        for obj_id in request.target_object_ids:
            obj = await self.object_repo.get_by_id(obj_id)
            if not obj or not obj.is_active:
                raise ObjectNotFoundError(obj_id)
        
        session = await self.session_repo.create(
            name=request.name,
            description=request.description,
            target_object_ids=request.target_object_ids,
            camera_source=request.camera_source
        )
        
        await self.db.commit()
        
        logger.info(f"Session created: {session.id} - {session.name}")
        
        # Get object names for response
        object_names = await self._get_object_names(request.target_object_ids)
        
        return SessionResponse(
            id=session.id,
            name=session.name,
            description=session.description,
            target_object_ids=session.target_object_ids,
            target_object_names=object_names,
            class_counts=session.class_counts or {},
            total_count=session.total_count,
            status=session.status,
            camera_source=session.camera_source,
            start_time=session.start_time,
            end_time=session.end_time,
            created_at=session.created_at
        )
    
    async def start_session(self, session_id: str) -> SessionResponse:
        """Start a counting session."""
        session = await self.session_repo.get_by_id(session_id)
        if not session:
            raise SessionNotFoundError(session_id)
        
        if session.status == "running":
            raise SessionAlreadyRunningError(session_id)
        
        # Initialize in-memory state
        object_names = await self._get_object_names(session.target_object_ids)
        self._active_sessions[session_id] = {
            "class_counts": {name: 0 for name in object_names.values()},
            "total_count": 0,
            "counted_ids": set(),
            "object_id_to_name": object_names
        }
        
        session = await self.session_repo.start_session(session_id)
        await self.db.commit()
        
        logger.info(f"Session started: {session_id}")
        
        return SessionResponse(
            id=session.id,
            name=session.name,
            description=session.description,
            target_object_ids=session.target_object_ids,
            target_object_names=object_names,
            class_counts=self._active_sessions[session_id]["class_counts"],
            total_count=0,
            status=session.status,
            camera_source=session.camera_source,
            start_time=session.start_time,
            end_time=session.end_time,
            created_at=session.created_at
        )
    
    async def stop_session(self, session_id: str) -> SessionStopResponse:
        """Stop a running session and save final counts."""
        session = await self.session_repo.get_by_id(session_id)
        if not session:
            raise SessionNotFoundError(session_id)
        
        if session.status != "running":
            raise SessionNotRunningError(session_id)
        
        # Get final counts from memory
        active_state = self._active_sessions.get(session_id, {})
        class_counts = active_state.get("class_counts", {})
        total_count = active_state.get("total_count", 0)
        
        session = await self.session_repo.stop_session(
            session_id=session_id,
            class_counts=class_counts,
            total_count=total_count
        )
        
        # Clean up in-memory state
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
        
        await self.db.commit()
        
        logger.info(f"Session stopped: {session_id}, total: {total_count}")
        
        return SessionStopResponse(
            id=session.id,
            name=session.name,
            status=session.status,
            class_counts=session.class_counts,
            total_count=session.total_count,
            start_time=session.start_time,
            end_time=session.end_time
        )
    
    async def get_session(self, session_id: str) -> SessionDetailResponse:
        """Get session details."""
        session = await self.session_repo.get_by_id(session_id)
        if not session:
            raise SessionNotFoundError(session_id)
        
        object_names = await self._get_object_names(session.target_object_ids)
        
        # Use live counts if session is running
        if session.status == "running" and session_id in self._active_sessions:
            class_counts = self._active_sessions[session_id]["class_counts"]
            total_count = self._active_sessions[session_id]["total_count"]
        else:
            class_counts = session.class_counts or {}
            total_count = session.total_count
        
        return SessionDetailResponse(
            id=session.id,
            name=session.name,
            description=session.description,
            target_object_ids=session.target_object_ids,
            target_object_names=object_names,
            class_counts=class_counts,
            total_count=total_count,
            status=session.status,
            camera_source=session.camera_source,
            start_time=session.start_time,
            end_time=session.end_time,
            created_at=session.created_at
        )
    
    async def list_sessions(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[SessionResponse]:
        """List all sessions."""
        sessions = await self.session_repo.get_all(
            status=status,
            limit=limit,
            offset=offset
        )
        
        result = []
        for session in sessions:
            object_names = await self._get_object_names(session.target_object_ids)
            
            # Use live counts for running sessions
            if session.status == "running" and session.id in self._active_sessions:
                class_counts = self._active_sessions[session.id]["class_counts"]
                total_count = self._active_sessions[session.id]["total_count"]
            else:
                class_counts = session.class_counts or {}
                total_count = session.total_count
            
            result.append(SessionResponse(
                id=session.id,
                name=session.name,
                description=session.description,
                target_object_ids=session.target_object_ids,
                target_object_names=object_names,
                class_counts=class_counts,
                total_count=total_count,
                status=session.status,
                camera_source=session.camera_source,
                start_time=session.start_time,
                end_time=session.end_time,
                created_at=session.created_at
            ))
        
        return result
    
    async def cancel_session(self, session_id: str) -> SessionResponse:
        """Cancel a session without saving results."""
        session = await self.session_repo.cancel_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)
        
        # Clean up in-memory state
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
        
        await self.db.commit()
        
        object_names = await self._get_object_names(session.target_object_ids)
        
        return SessionResponse(
            id=session.id,
            name=session.name,
            description=session.description,
            target_object_ids=session.target_object_ids,
            target_object_names=object_names,
            class_counts=session.class_counts or {},
            total_count=session.total_count,
            status=session.status,
            camera_source=session.camera_source,
            start_time=session.start_time,
            end_time=session.end_time,
            created_at=session.created_at
        )
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        session = await self.session_repo.get_by_id(session_id)
        if not session:
            raise SessionNotFoundError(session_id)
        
        if session.status == "running":
            raise SessionAlreadyRunningError(session_id)
        
        success = await self.session_repo.delete(session_id)
        await self.db.commit()
        return success
    
    # ==========================================
    # Real-time counting methods
    # ==========================================
    
    def update_count(
        self, 
        session_id: str, 
        object_name: str, 
        track_id: int
    ) -> Dict:
        """
        Update count for a session (called by counting pipeline).
        Returns current counts.
        """
        if session_id not in self._active_sessions:
            return {}
        
        state = self._active_sessions[session_id]
        
        # Only count if this track hasn't been counted
        if track_id not in state["counted_ids"]:
            state["counted_ids"].add(track_id)
            state["total_count"] += 1
            
            if object_name in state["class_counts"]:
                state["class_counts"][object_name] += 1
            else:
                state["class_counts"][object_name] = 1
        
        return {
            "class_counts": state["class_counts"],
            "total_count": state["total_count"]
        }
    
    def get_live_counts(self, session_id: str) -> Optional[Dict]:
        """Get current counts for an active session."""
        if session_id not in self._active_sessions:
            return None
        
        state = self._active_sessions[session_id]
        return {
            "class_counts": state["class_counts"],
            "total_count": state["total_count"]
        }
    
    def is_session_active(self, session_id: str) -> bool:
        """Check if session is currently active in memory."""
        return session_id in self._active_sessions
    
    async def _get_object_names(self, object_ids: List[str]) -> Dict[str, str]:
        """Get object names from IDs. Returns {object_id: object_name}"""
        names = {}
        for obj_id in object_ids:
            obj = await self.object_repo.get_by_id(obj_id)
            if obj:
                names[obj_id] = obj.name
        return names