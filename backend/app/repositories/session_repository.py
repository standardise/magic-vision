from typing import List, Optional, Dict
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from app.db.models.session import CountingSession


class SessionRepository:
    """Repository for CountingSession CRUD operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create(
        self,
        name: str,
        target_object_ids: List[str],
        description: Optional[str] = None,
        camera_source: Optional[str] = None
    ) -> CountingSession:
        """Create a new counting session."""
        session = CountingSession(
            name=name,
            description=description,
            target_object_ids=target_object_ids,
            camera_source=camera_source,
            status="created",
            class_counts={},
            total_count=0
        )
        self.db.add(session)
        await self.db.flush()
        return session
    
    async def get_by_id(self, session_id: str) -> Optional[CountingSession]:
        """Get session by ID."""
        result = await self.db.execute(
            select(CountingSession).where(CountingSession.id == session_id)
        )
        return result.scalar_one_or_none()
    
    async def get_all(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[CountingSession]:
        """Get all sessions with optional status filter."""
        query = select(CountingSession).limit(limit).offset(offset).order_by(CountingSession.created_at.desc())
        if status:
            query = query.where(CountingSession.status == status)
        result = await self.db.execute(query)
        return list(result.scalars().all())
    
    async def get_running_sessions(self) -> List[CountingSession]:
        """Get all currently running sessions."""
        result = await self.db.execute(
            select(CountingSession).where(CountingSession.status == "running")
        )
        return list(result.scalars().all())
    
    async def start_session(self, session_id: str) -> Optional[CountingSession]:
        """Start a session (set status to running)."""
        session = await self.get_by_id(session_id)
        if not session:
            return None
        
        session.status = "running"
        session.start_time = datetime.utcnow()
        await self.db.flush()
        return session
    
    async def stop_session(
        self, 
        session_id: str,
        class_counts: Dict[str, int],
        total_count: int
    ) -> Optional[CountingSession]:
        """Stop a session and save final counts."""
        session = await self.get_by_id(session_id)
        if not session:
            return None
        
        session.status = "stopped"
        session.end_time = datetime.utcnow()
        session.class_counts = class_counts
        session.total_count = total_count
        await self.db.flush()
        return session
    
    async def update_counts(
        self,
        session_id: str,
        class_counts: Dict[str, int],
        total_count: int
    ) -> bool:
        """Update session counts (real-time update)."""
        result = await self.db.execute(
            update(CountingSession)
            .where(CountingSession.id == session_id)
            .values(class_counts=class_counts, total_count=total_count)
        )
        return result.rowcount > 0
    
    async def cancel_session(self, session_id: str) -> Optional[CountingSession]:
        """Cancel a session."""
        session = await self.get_by_id(session_id)
        if not session:
            return None
        
        session.status = "cancelled"
        session.end_time = datetime.utcnow()
        await self.db.flush()
        return session
    
    async def delete(self, session_id: str) -> bool:
        """Delete a session (hard delete)."""
        from sqlalchemy import delete
        result = await self.db.execute(
            delete(CountingSession).where(CountingSession.id == session_id)
        )
        return result.rowcount > 0
