from minio import Minio
from minio.error import S3Error
from io import BytesIO
from typing import Optional, List
from uuid import uuid4
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class MinioClient:
    """MinIO storage client wrapper."""
    
    def __init__(self):
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.MINIO_ROOT_USER,
            secret_key=settings.MINIO_ROOT_PASSWORD,
            secure=settings.MINIO_SECURE
        )
        self.bucket = settings.MINIO_BUCKET
        self._ensure_bucket()
    
    def _ensure_bucket(self) -> None:
        """Create bucket if it doesn't exist."""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info(f"Created bucket: {self.bucket}")
        except S3Error as e:
            logger.error(f"Failed to ensure bucket: {e}")
            raise
    
    async def upload_image(
        self, 
        file_data: bytes, 
        object_id: str,
        filename: str,
        content_type: str = "image/jpeg"
    ) -> str:
        """
        Upload image to MinIO.
        Returns the object path in MinIO.
        """
        # Generate unique filename
        ext = filename.rsplit(".", 1)[-1] if "." in filename else "jpg"
        unique_name = f"{uuid4().hex}.{ext}"
        object_path = f"objects/{object_id}/{unique_name}"
        
        try:
            data_stream = BytesIO(file_data)
            self.client.put_object(
                bucket_name=self.bucket,
                object_name=object_path,
                data=data_stream,
                length=len(file_data),
                content_type=content_type
            )
            logger.info(f"Uploaded: {object_path}")
            return object_path
        except S3Error as e:
            logger.error(f"Upload failed: {e}")
            raise
    
    async def get_image(self, object_path: str) -> Optional[bytes]:
        """Download image from MinIO."""
        try:
            response = self.client.get_object(self.bucket, object_path)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            logger.error(f"Download failed: {e}")
            return None
    
    async def delete_image(self, object_path: str) -> bool:
        """Delete image from MinIO."""
        try:
            self.client.remove_object(self.bucket, object_path)
            logger.info(f"Deleted: {object_path}")
            return True
        except S3Error as e:
            logger.error(f"Delete failed: {e}")
            return False
    
    async def delete_folder(self, folder_prefix: str) -> int:
        """Delete all objects in a folder. Returns count of deleted objects."""
        deleted_count = 0
        try:
            objects = self.client.list_objects(self.bucket, prefix=folder_prefix, recursive=True)
            for obj in objects:
                self.client.remove_object(self.bucket, obj.object_name)
                deleted_count += 1
            logger.info(f"Deleted {deleted_count} objects from {folder_prefix}")
        except S3Error as e:
            logger.error(f"Folder delete failed: {e}")
        return deleted_count
    
    def get_presigned_url(self, object_path: str, expires_hours: int = 24) -> str:
        """Get presigned URL for image access."""
        from datetime import timedelta
        try:
            url = self.client.presigned_get_object(
                self.bucket,
                object_path,
                expires=timedelta(hours=expires_hours)
            )
            return url
        except S3Error as e:
            logger.error(f"Failed to generate URL: {e}")
            return ""
    
    def list_images(self, object_id: str) -> List[str]:
        """List all images for an object."""
        prefix = f"objects/{object_id}/"
        paths = []
        try:
            objects = self.client.list_objects(self.bucket, prefix=prefix)
            for obj in objects:
                paths.append(obj.object_name)
        except S3Error as e:
            logger.error(f"List failed: {e}")
        return paths


# Singleton instance
_minio_client: Optional[MinioClient] = None


def get_minio_client() -> MinioClient:
    """Get MinIO client singleton."""
    global _minio_client
    if _minio_client is None:
        _minio_client = MinioClient()
    return _minio_client
