from pydantic import BaseModel
from typing import Dict, Optional, List
from enum import Enum


class StreamMessageType(str, Enum):
    """Types of WebSocket messages."""
    # Client -> Server
    FRAME = "frame"           # Video frame (base64)
    CAMERA_SWITCH = "camera_switch"  # Switch camera source
    LINE_CONFIG = "line_config"      # Configure counting line
    STOP = "stop"             # Stop streaming
    
    # Server -> Client
    RESULT = "result"         # Processed frame + counts
    ERROR = "error"           # Error message
    STATUS = "status"         # Session status update


class StreamFrame(BaseModel):
    """Video frame from client."""
    type: str = StreamMessageType.FRAME
    data: str  # Base64 encoded image
    timestamp: Optional[float] = None


class StreamResult(BaseModel):
    """Processing result sent to client."""
    type: str = StreamMessageType.RESULT
    frame: str  # Base64 encoded annotated frame
    detections: int
    class_counts: Dict[str, int]
    total_count: int
    fps: float
    timestamp: float


class StreamError(BaseModel):
    """Error message."""
    type: str = StreamMessageType.ERROR
    message: str
    code: Optional[str] = None


class StreamStatus(BaseModel):
    """Session status update."""
    type: str = StreamMessageType.STATUS
    status: str
    session_id: str
    message: Optional[str] = None


class CameraSwitchRequest(BaseModel):
    """Request to switch camera source."""
    type: str = StreamMessageType.CAMERA_SWITCH
    source: str  # "webcam", "rtsp://...", "usb:0"


class LineConfigRequest(BaseModel):
    """Configure counting line."""
    type: str = StreamMessageType.LINE_CONFIG
    start_point: List[int]  # [x1, y1]
    end_point: List[int]    # [x2, y2]
