import numpy as np
import cv2
from typing import List, Optional, Dict
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.repositories.object_repository import ObjectRepository
from app.storage.minio_client import MinioClient
from app.ml.encoder import DINOv2Encoder
from app.core.exceptions import ObjectNotFoundError, InsufficientImagesError, InvalidImageError
from app.schemas.object import ObjectResponse, ObjectDetailResponse

logger = logging.getLogger(__name__)


class ObjectService:
    """Business logic for Object Library operations."""
    
    MIN_IMAGES = 3
    MAX_IMAGES = 10
    
    def __init__(
        self, 
        db: AsyncSession, 
        dino_encoder: DINOv2Encoder, 
        minio_client: MinioClient
    ):
        self.repo = ObjectRepository(db)
        self.dino = dino_encoder
        self.minio = minio_client
        self.db = db
    
    async def create_object(
        self,
        name: str,
        description: Optional[str],
        files: List[UploadFile]
    ) -> ObjectResponse:
        """
        Create a new object with reference images.
        1. Validate image count (3-10)
        2. Create object record
        3. Upload images to MinIO
        4. Encode images with DINOv2
        5. Calculate prototype embedding (mean of all embeddings)
        6. Store embedding in pgvector
        """
        # Validate image count
        if len(files) < self.MIN_IMAGES or len(files) > self.MAX_IMAGES:
            raise InsufficientImagesError(self.MIN_IMAGES, self.MAX_IMAGES)
        
        # Create object record
        obj = await self.repo.create(name=name, description=description)
        logger.info(f"Created object: {obj.id} - {name}")
        
        vectors = []
        image_paths = []
        
        # Process each image
        for file in files:
            try:
                # Read image bytes
                content = await file.read()
                await file.seek(0)
                
                # Validate it's an image
                image_array = self._bytes_to_cv2(content)
                if image_array is None:
                    raise InvalidImageError(f"Failed to decode: {file.filename}")
                
                # Upload to MinIO
                image_path = await self.minio.upload_image(
                    file_data=content,
                    object_id=obj.id,
                    filename=file.filename or "image.jpg",
                    content_type=file.content_type or "image/jpeg"
                )
                image_paths.append(image_path)
                
                # Save image record
                await self.repo.add_image(obj.id, image_path)
                
                # Encode with DINOv2
                vector = self.dino.encode(image_array)
                vectors.append(vector)
                
            except Exception as e:
                logger.error(f"Failed to process image {file.filename}: {e}")
                # Cleanup on error
                await self.minio.delete_folder(f"objects/{obj.id}/")
                await self.repo.hard_delete(obj.id)
                raise InvalidImageError(str(e))
        
        # Calculate prototype embedding (mean of all vectors)
        prototype_vector = np.mean(vectors, axis=0)
        
        # Normalize the prototype vector
        prototype_vector = prototype_vector / (np.linalg.norm(prototype_vector) + 1e-10)
        
        # Save embedding
        await self.repo.update_embedding(obj.id, prototype_vector)
        await self.db.commit()
        
        logger.info(f"Object {obj.id} created with {len(vectors)} images")
        
        # Generate presigned URLs for response
        image_urls = [self.minio.get_presigned_url(p) for p in image_paths]
        
        return ObjectResponse(
            id=obj.id,
            name=obj.name,
            description=obj.description,
            image_urls=image_urls,
            image_count=len(image_paths),
            is_active=obj.is_active,
            created_at=obj.created_at,
            updated_at=obj.updated_at
        )
    
    async def get_object(self, object_id: str) -> ObjectDetailResponse:
        """Get object details with images."""
        obj = await self.repo.get_by_id(object_id, include_images=True)
        if not obj:
            raise ObjectNotFoundError(object_id)
        
        image_urls = [self.minio.get_presigned_url(img.image_path) for img in obj.images]
        
        return ObjectDetailResponse(
            id=obj.id,
            name=obj.name,
            description=obj.description,
            image_urls=image_urls,
            image_count=len(obj.images),
            is_active=obj.is_active,
            has_embedding=obj.embedding is not None,
            created_at=obj.created_at,
            updated_at=obj.updated_at
        )
    
    async def list_objects(
        self, 
        active_only: bool = True,
        limit: int = 100,
        offset: int = 0
    ) -> List[ObjectResponse]:
        """List all objects."""
        objects = await self.repo.get_all(
            active_only=active_only,
            include_images=True,
            limit=limit,
            offset=offset
        )
        
        return [
            ObjectResponse(
                id=obj.id,
                name=obj.name,
                description=obj.description,
                image_urls=[self.minio.get_presigned_url(img.image_path) for img in obj.images],
                image_count=len(obj.images),
                is_active=obj.is_active,
                created_at=obj.created_at,
                updated_at=obj.updated_at
            )
            for obj in objects
        ]
    
    async def update_object(
        self,
        object_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> ObjectResponse:
        """Update object metadata."""
        obj = await self.repo.update(object_id, name=name, description=description)
        if not obj:
            raise ObjectNotFoundError(object_id)
        
        await self.db.commit()
        
        images = await self.repo.get_images(object_id)
        image_urls = [self.minio.get_presigned_url(img.image_path) for img in images]
        
        return ObjectResponse(
            id=obj.id,
            name=obj.name,
            description=obj.description,
            image_urls=image_urls,
            image_count=len(images),
            is_active=obj.is_active,
            created_at=obj.created_at,
            updated_at=obj.updated_at
        )
    
    async def add_images(
        self,
        object_id: str,
        files: List[UploadFile]
    ) -> ObjectResponse:
        """Add more reference images to an existing object."""
        obj = await self.repo.get_by_id(object_id, include_images=True)
        if not obj:
            raise ObjectNotFoundError(object_id)
        
        current_count = len(obj.images)
        if current_count + len(files) > self.MAX_IMAGES:
            raise InsufficientImagesError(
                self.MIN_IMAGES, 
                self.MAX_IMAGES
            )
        
        all_vectors = []
        
        # Get existing image vectors
        for img in obj.images:
            img_data = await self.minio.get_image(img.image_path)
            if img_data:
                image_array = self._bytes_to_cv2(img_data)
                if image_array is not None:
                    vector = self.dino.encode(image_array)
                    all_vectors.append(vector)
        
        # Process new images
        for file in files:
            content = await file.read()
            image_array = self._bytes_to_cv2(content)
            if image_array is None:
                raise InvalidImageError(f"Failed to decode: {file.filename}")
            
            # Upload to MinIO
            image_path = await self.minio.upload_image(
                file_data=content,
                object_id=object_id,
                filename=file.filename or "image.jpg",
                content_type=file.content_type or "image/jpeg"
            )
            
            await self.repo.add_image(object_id, image_path)
            
            vector = self.dino.encode(image_array)
            all_vectors.append(vector)
        
        # Recalculate prototype embedding
        prototype_vector = np.mean(all_vectors, axis=0)
        prototype_vector = prototype_vector / (np.linalg.norm(prototype_vector) + 1e-10)
        await self.repo.update_embedding(object_id, prototype_vector)
        
        await self.db.commit()
        
        # Refresh object
        return await self.get_object(object_id)
    
    async def delete_object(self, object_id: str, hard: bool = False) -> bool:
        """Delete an object (soft or hard delete)."""
        obj = await self.repo.get_by_id(object_id)
        if not obj:
            raise ObjectNotFoundError(object_id)
        
        if hard:
            # Delete images from MinIO
            await self.minio.delete_folder(f"objects/{object_id}/")
            success = await self.repo.hard_delete(object_id)
        else:
            success = await self.repo.soft_delete(object_id)
        
        await self.db.commit()
        return success
    
    async def reactivate_object(self, object_id: str) -> ObjectResponse:
        """Reactivate a soft-deleted object."""
        success = await self.repo.reactivate(object_id)
        if not success:
            raise ObjectNotFoundError(object_id)
        
        await self.db.commit()
        return await self.get_object(object_id)
    
    async def get_embeddings_for_session(
        self, 
        object_ids: List[str]
    ) -> Dict[str, Dict]:
        """
        Get embeddings for specific objects.
        Used when starting a counting session.
        Returns: {object_id: {"name": str, "embedding": np.ndarray}}
        """
        return await self.repo.get_embeddings_by_ids(object_ids)
    
    def _bytes_to_cv2(self, data: bytes) -> Optional[np.ndarray]:
        """Convert bytes to OpenCV image (BGR)."""
        try:
            nparr = np.frombuffer(data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return image
        except Exception:
            return None