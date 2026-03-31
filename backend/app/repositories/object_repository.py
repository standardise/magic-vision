from typing import List, Optional, Dict
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from pgvector.sqlalchemy import Vector
import numpy as np

from app.db.models.object import Object, ObjectImage
from app.core.config import settings


class ObjectRepository:
    """Repository for Object CRUD operations with pgvector support."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create(self, name: str, description: Optional[str] = None) -> Object:
        """Create a new object without embedding."""
        obj = Object(name=name, description=description)
        self.db.add(obj)
        await self.db.flush()
        return obj
    
    async def get_by_id(self, object_id: str, include_images: bool = False) -> Optional[Object]:
        """Get object by ID."""
        query = select(Object).where(Object.id == object_id)
        if include_images:
            query = query.options(selectinload(Object.images))
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_all(
        self, 
        active_only: bool = True,
        include_images: bool = False,
        limit: int = 100,
        offset: int = 0
    ) -> List[Object]:
        """Get all objects with pagination."""
        query = select(Object).limit(limit).offset(offset).order_by(Object.created_at.desc())
        if active_only:
            query = query.where(Object.is_active == True)
        if include_images:
            query = query.options(selectinload(Object.images))
        result = await self.db.execute(query)
        return list(result.scalars().all())
    
    async def update(
        self, 
        object_id: str, 
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Optional[Object]:
        """Update object metadata."""
        obj = await self.get_by_id(object_id)
        if not obj:
            return None
        if name is not None:
            obj.name = name
        if description is not None:
            obj.description = description
        await self.db.flush()
        return obj
    
    async def update_embedding(self, object_id: str, embedding: np.ndarray) -> bool:
        """Update object embedding vector."""
        result = await self.db.execute(
            update(Object)
            .where(Object.id == object_id)
            .values(embedding=embedding.tolist())
        )
        return result.rowcount > 0
    
    async def soft_delete(self, object_id: str) -> bool:
        """Soft delete (deactivate) an object."""
        result = await self.db.execute(
            update(Object)
            .where(Object.id == object_id)
            .values(is_active=False)
        )
        return result.rowcount > 0
    
    async def hard_delete(self, object_id: str) -> bool:
        """Permanently delete an object and its images."""
        result = await self.db.execute(
            delete(Object).where(Object.id == object_id)
        )
        return result.rowcount > 0
    
    async def reactivate(self, object_id: str) -> bool:
        """Reactivate a soft-deleted object."""
        result = await self.db.execute(
            update(Object)
            .where(Object.id == object_id)
            .values(is_active=True)
        )
        return result.rowcount > 0
    
    # ==========================================
    # Image Operations
    # ==========================================
    
    async def add_image(self, object_id: str, image_path: str) -> ObjectImage:
        """Add reference image to object."""
        img = ObjectImage(object_id=object_id, image_path=image_path)
        self.db.add(img)
        await self.db.flush()
        return img
    
    async def get_images(self, object_id: str) -> List[ObjectImage]:
        """Get all images for an object."""
        result = await self.db.execute(
            select(ObjectImage).where(ObjectImage.object_id == object_id)
        )
        return list(result.scalars().all())
    
    async def delete_image(self, image_id: str) -> bool:
        """Delete a specific image."""
        result = await self.db.execute(
            delete(ObjectImage).where(ObjectImage.id == image_id)
        )
        return result.rowcount > 0
    
    # ==========================================
    # Vector Search Operations
    # ==========================================
    
    async def find_similar(
        self, 
        query_vector: np.ndarray, 
        limit: int = 5,
        threshold: float = None
    ) -> List[tuple]:
        """
        Find similar objects using cosine similarity.
        Returns list of (Object, similarity_score) tuples.
        """
        threshold = threshold or settings.SIMILARITY_THRESHOLD
        
        # pgvector cosine distance: <=> operator
        # similarity = 1 - distance
        query = (
            select(
                Object,
                (1 - Object.embedding.cosine_distance(query_vector.tolist())).label("similarity")
            )
            .where(Object.is_active == True)
            .where(Object.embedding.isnot(None))
            .order_by(Object.embedding.cosine_distance(query_vector.tolist()))
            .limit(limit)
        )
        
        result = await self.db.execute(query)
        rows = result.all()
        
        # Filter by threshold
        return [(obj, sim) for obj, sim in rows if sim >= threshold]
    
    async def get_embeddings_by_ids(self, object_ids: List[str]) -> Dict[str, np.ndarray]:
        """
        Get embeddings for specific objects.
        Returns dict: {object_id: embedding_vector}
        """
        result = await self.db.execute(
            select(Object.id, Object.name, Object.embedding)
            .where(Object.id.in_(object_ids))
            .where(Object.embedding.isnot(None))
        )
        rows = result.all()
        
        return {
            row.id: {
                "name": row.name,
                "embedding": np.array(row.embedding) if row.embedding else None
            }
            for row in rows
        }
    
    async def get_all_embeddings(self, active_only: bool = True) -> Dict[str, Dict]:
        """Get all embeddings for active objects."""
        query = select(Object.id, Object.name, Object.embedding).where(Object.embedding.isnot(None))
        if active_only:
            query = query.where(Object.is_active == True)
        
        result = await self.db.execute(query)
        rows = result.all()
        
        return {
            row.id: {
                "name": row.name,
                "embedding": np.array(row.embedding)
            }
            for row in rows
        }
