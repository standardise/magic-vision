import cv2
import base64
import json
import numpy as np
import asyncio
import logging
from typing import Optional, Tuple
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from datetime import datetime

from app.services.counting_service import CountingService
from app.services.session_service import SessionService
from app.repositories.object_repository import ObjectRepository
from app.db.session import AsyncSessionLocal
from app.schemas.stream import StreamMessageType
from app.core.exceptions import SessionNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stream", tags=["Stream"])


class ConnectionManager:
    """Manages WebSocket connections for real-time streaming."""
    
    def __init__(self):
        # {session_id: WebSocket}
        self.active_connections: dict = {}
    
    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected for session {session_id}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected for session {session_id}")
    
    async def send_json(self, session_id: str, data: dict):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(data)
            except Exception as e:
                logger.error(f"Failed to send to {session_id}: {e}")


manager = ConnectionManager()


def decode_base64_image(data: str) -> Optional[np.ndarray]:
    """Decode base64 string to OpenCV image."""
    try:
        # Remove data URL prefix if present
        if "," in data:
            data = data.split(",")[1]
        
        img_bytes = base64.b64decode(data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        return None


def encode_image_base64(image: np.ndarray, quality: int = 80) -> str:
    """Encode OpenCV image to base64 string."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buffer = cv2.imencode('.jpg', image, encode_param)
    return base64.b64encode(buffer).decode('utf-8')


@router.websocket("/{session_id}")
async def stream_websocket(
    websocket: WebSocket,
    session_id: str
):
    """
    WebSocket endpoint for real-time video streaming and counting.
    
    Client sends:
    - {"type": "frame", "data": "<base64 image>"}
    - {"type": "line_config", "start_point": [x1,y1], "end_point": [x2,y2]}
    - {"type": "stop"}
    
    Server sends:
    - {"type": "result", "frame": "<base64>", "class_counts": {...}, "total_count": N, "fps": X}
    - {"type": "error", "message": "..."}
    - {"type": "status", "status": "...", "session_id": "..."}
    """
    await manager.connect(session_id, websocket)
    
    # Default counting line (horizontal, middle of frame)
    counting_line: Optional[Tuple] = None
    counting_service: Optional[CountingService] = None
    
    try:
        # Initialize services with new DB session
        async with AsyncSessionLocal() as db:
            # Get session info and prototypes
            session_service = SessionService(db)
            object_repo = ObjectRepository(db)
            
            try:
                session = await session_service.get_session(session_id)
            except SessionNotFoundError:
                await websocket.send_json({
                    "type": StreamMessageType.ERROR.value,
                    "message": f"Session {session_id} not found"
                })
                await websocket.close()
                return
            
            # Get embeddings for target objects
            prototypes = await object_repo.get_embeddings_by_ids(session.target_object_ids)
            
            if not prototypes:
                await websocket.send_json({
                    "type": StreamMessageType.ERROR.value,
                    "message": "No valid target objects with embeddings"
                })
                await websocket.close()
                return
            
            # Initialize ML models and counting service
            from app.api.v1.deps import get_yolo_model, get_dino_encoder
            yolo = get_yolo_model()
            dino = get_dino_encoder()
            
            counting_service = CountingService(
                yolo=yolo,
                dino=dino,
                session_service=session_service
            )
            
            await counting_service.initialize_session(session_id, prototypes)
            
            # Start session if not already running
            if session.status != "running":
                await session_service.start_session(session_id)
                await db.commit()
            
            await websocket.send_json({
                "type": StreamMessageType.STATUS.value,
                "status": "ready",
                "session_id": session_id,
                "message": "Counting pipeline initialized"
            })
            
            # Main processing loop
            while True:
                try:
                    # Receive message with timeout
                    data = await websocket.receive_json()
                    msg_type = data.get("type", "")
                    
                    if msg_type == StreamMessageType.FRAME.value:
                        # Process video frame
                        image = decode_base64_image(data.get("data", ""))
                        if image is None:
                            continue
                        
                        # Set default counting line if not configured
                        if counting_line is None:
                            h, w = image.shape[:2]
                            counting_line = ((50, h // 2), (w - 50, h // 2))
                        
                        # Process frame
                        result = await counting_service.process_frame(
                            session_id=session_id,
                            frame=image,
                            counting_line=counting_line
                        )
                        
                        if result:
                            # Send result back
                            await websocket.send_json({
                                "type": StreamMessageType.RESULT.value,
                                "frame": encode_image_base64(result.frame),
                                "detections": result.detections,
                                "class_counts": result.class_counts,
                                "total_count": result.total_count,
                                "fps": round(result.fps, 1),
                                "timestamp": datetime.now().timestamp()
                            })
                    
                    elif msg_type == StreamMessageType.LINE_CONFIG.value:
                        # Configure counting line
                        start = data.get("start_point", [])
                        end = data.get("end_point", [])
                        if len(start) == 2 and len(end) == 2:
                            counting_line = (tuple(start), tuple(end))
                            await websocket.send_json({
                                "type": StreamMessageType.STATUS.value,
                                "status": "line_configured",
                                "session_id": session_id,
                                "message": f"Counting line set: {counting_line}"
                            })
                    
                    elif msg_type == StreamMessageType.STOP.value:
                        # Stop session
                        await session_service.stop_session(session_id)
                        await db.commit()
                        if counting_service:
                            await counting_service.cleanup_session(session_id)
                        await websocket.send_json({
                            "type": StreamMessageType.STATUS.value,
                            "status": "stopped",
                            "session_id": session_id
                        })
                        break
                    
                except WebSocketDisconnect:
                    logger.info(f"WebSocket disconnected: {session_id}")
                    break
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON received")
                    continue
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await websocket.send_json({
                        "type": StreamMessageType.ERROR.value,
                        "message": str(e)
                    })
            
            # Cleanup
            if counting_service:
                await counting_service.cleanup_session(session_id)
    
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        manager.disconnect(session_id)


@router.get("/test")
async def test_stream():
    """Test endpoint to verify stream router is working."""
    return {"status": "ok", "message": "Stream endpoint ready"}
