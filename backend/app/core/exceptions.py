from fastapi import HTTPException, status


class ObjectNotFoundError(HTTPException):
    def __init__(self, object_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Object with id '{object_id}' not found"
        )


class SessionNotFoundError(HTTPException):
    def __init__(self, session_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with id '{session_id}' not found"
        )


class SessionAlreadyRunningError(HTTPException):
    def __init__(self, session_id: str):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Session '{session_id}' is already running"
        )


class SessionNotRunningError(HTTPException):
    def __init__(self, session_id: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session '{session_id}' is not running"
        )


class InvalidImageError(HTTPException):
    def __init__(self, message: str = "Invalid image file"):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )


class MinIOConnectionError(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to connect to storage service"
        )


class InsufficientImagesError(HTTPException):
    def __init__(self, min_images: int = 3, max_images: int = 10):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Please provide between {min_images} and {max_images} reference images"
        )


class NoTargetObjectsError(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please specify at least one target object to count"
        )
