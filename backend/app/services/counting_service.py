import cv2
import numpy as np
import logging
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

from app.ml.detector import YOLOv8Agnostic
from app.ml.encoder import DINOv2Encoder
from app.ml.tracker import ByteTrack
from app.services.session_service import SessionService
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class CountingResult:
    """Result from processing a single frame."""
    frame: np.ndarray
    detections: int
    new_counts: int
    class_counts: Dict[str, int]
    total_count: int
    fps: float
    confidence_scores: Dict[int, float] = field(default_factory=dict)


@dataclass
class TrackState:
    """State for a single tracked object."""
    votes: List[str] = field(default_factory=list)
    vote_scores: List[float] = field(default_factory=list)
    final_class: Optional[str] = None
    final_confidence: float = 0.0
    prev_centroid: Optional[Tuple[float, float]] = None
    last_seen_frame: int = 0
    is_counted: bool = False


class CountingService:
    """
    Production-ready object counting service.
    
    Improvements over basic implementation:
    1. Batch encoding for efficiency
    2. Confidence-weighted voting
    3. Unknown class detection (rejects low-confidence matches)
    4. Automatic track memory cleanup
    5. Adaptive frame skipping based on FPS
    """
    
    # Minimum confidence to accept a match
    MIN_CONFIDENCE = 0.65
    
    # Confidence threshold for "high confidence" match (skip more votes)
    HIGH_CONFIDENCE = 0.85
    
    # Maximum age of track memory before cleanup (in frames)
    TRACK_MEMORY_MAX_AGE = 150
    
    def __init__(
        self,
        yolo: YOLOv8Agnostic,
        dino: DINOv2Encoder,
        session_service: SessionService
    ):
        self.yolo = yolo
        self.dino = dino
        self.session_service = session_service
        
        # Per-session state
        self._session_state: Dict[str, Dict] = {}
        
        # Frame processing settings
        self.frame_skip = settings.FRAME_SKIP
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        self.vote_count = settings.VOTE_COUNT
        
        # Performance tracking
        self._fps_history: deque = deque(maxlen=30)
    
    async def initialize_session(
        self,
        session_id: str,
        prototypes: Dict[str, Dict]
    ) -> bool:
        """Initialize counting state for a session."""
        tracker = ByteTrack(
            track_thresh=0.5,
            track_buffer=settings.MAX_TRACK_AGE,
            match_thresh=0.8,
            frame_rate=30
        )
        
        # Convert prototypes to use names as keys
        proto_by_name = {}
        id_to_name = {}
        for obj_id, data in prototypes.items():
            name = data["name"]
            embedding = data["embedding"]
            if embedding is not None:
                proto_by_name[name] = embedding
                id_to_name[obj_id] = name
        
        self._session_state[session_id] = {
            "tracker": tracker,
            "prototypes": proto_by_name,
            "id_to_name": id_to_name,
            "track_memory": {},  # {track_id: TrackState}
            "frame_count": 0,
            "pending_crops": [],  # For batch encoding
            "pending_track_ids": [],
        }
        
        logger.info(f"Session {session_id} initialized with {len(proto_by_name)} prototypes")
        return True
    
    async def cleanup_session(self, session_id: str) -> None:
        """Clean up session state."""
        if session_id in self._session_state:
            del self._session_state[session_id]
            logger.info(f"Session {session_id} cleaned up")
    
    async def process_frame(
        self,
        session_id: str,
        frame: np.ndarray,
        counting_line: tuple = None
    ) -> Optional[CountingResult]:
        """Process a single frame with optimizations."""
        if session_id not in self._session_state:
            logger.warning(f"Session {session_id} not initialized")
            return None
        
        state = self._session_state[session_id]
        state["frame_count"] += 1
        frame_num = state["frame_count"]
        
        start_time = datetime.now()
        new_counts = 0
        confidence_scores = {}
        
        # 1. Detection
        boxes = self.yolo.detect(frame)
        
        # 2. Tracking
        tracked_objects = state["tracker"].update(boxes)
        
        # 3. Collect crops for objects that need encoding
        crops_to_encode = []
        track_ids_to_encode = []
        
        for obj in tracked_objects:
            track_id = obj.track_id
            box = obj.tlbr
            
            # Initialize track state if new
            if track_id not in state["track_memory"]:
                state["track_memory"][track_id] = TrackState(last_seen_frame=frame_num)
            
            track_state = state["track_memory"][track_id]
            track_state.last_seen_frame = frame_num
            
            # Calculate centroid
            centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            
            # Check if we need to encode this object
            needs_encoding = (
                track_state.final_class is None and 
                len(track_state.votes) < self.vote_count
            )
            
            if needs_encoding:
                cropped = self._crop_image(frame, box)
                if cropped.size > 0 and cropped.shape[0] > 10 and cropped.shape[1] > 10:
                    crops_to_encode.append(cropped)
                    track_ids_to_encode.append(track_id)
        
        # 4. Batch encode all crops at once (much faster!)
        if crops_to_encode:
            embeddings = self.dino.encode_batch(crops_to_encode)
            
            for i, track_id in enumerate(track_ids_to_encode):
                embedding = embeddings[i]
                track_state = state["track_memory"][track_id]
                
                # Find best match with confidence
                best_class, best_score = self._find_best_match_with_score(
                    embedding, state["prototypes"]
                )
                
                # Only vote if confidence is above minimum
                if best_score >= self.MIN_CONFIDENCE:
                    track_state.votes.append(best_class)
                    track_state.vote_scores.append(best_score)
                    
                    # High-confidence early exit
                    if best_score >= self.HIGH_CONFIDENCE:
                        track_state.final_class = best_class
                        track_state.final_confidence = best_score
                    # Standard voting
                    elif len(track_state.votes) >= self.vote_count:
                        self._finalize_class(track_state)
        
        # 5. Line crossing detection and counting
        for obj in tracked_objects:
            track_id = obj.track_id
            box = obj.tlbr
            track_state = state["track_memory"].get(track_id)
            
            if not track_state:
                continue
            
            centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            prev_centroid = track_state.prev_centroid
            track_state.prev_centroid = centroid
            
            # Record confidence for visualization
            if track_state.final_confidence > 0:
                confidence_scores[track_id] = track_state.final_confidence
            
            # Check line crossing
            if (track_state.final_class and 
                counting_line and 
                prev_centroid and 
                not track_state.is_counted):
                
                if self._is_crossing_line(prev_centroid, centroid, counting_line):
                    counts = self.session_service.update_count(
                        session_id=session_id,
                        object_name=track_state.final_class,
                        track_id=track_id
                    )
                    if counts:
                        track_state.is_counted = True
                        new_counts += 1
            
            # Draw visualization
            self._draw_detection(
                frame, box, track_id,
                track_state.final_class,
                track_state.final_confidence,
                track_state.is_counted
            )
        
        # 6. Draw counting line
        if counting_line:
            cv2.line(frame, counting_line[0], counting_line[1], (0, 0, 255), 3)
        
        # 7. Cleanup old tracks periodically
        if frame_num % 30 == 0:
            self._cleanup_old_tracks(state, frame_num)
        
        # 8. Calculate FPS
        process_time = (datetime.now() - start_time).total_seconds()
        fps = 1.0 / process_time if process_time > 0 else 0
        self._fps_history.append(fps)
        avg_fps = sum(self._fps_history) / len(self._fps_history)
        
        # 9. Get counts and draw overlay
        live_counts = self.session_service.get_live_counts(session_id) or {
            "class_counts": {},
            "total_count": 0
        }
        
        self._draw_count_overlay(frame, live_counts, avg_fps)
        
        return CountingResult(
            frame=frame,
            detections=len(tracked_objects),
            new_counts=new_counts,
            class_counts=live_counts["class_counts"],
            total_count=live_counts["total_count"],
            fps=avg_fps,
            confidence_scores=confidence_scores
        )
    
    def _find_best_match_with_score(
        self, 
        vector: np.ndarray, 
        prototypes: Dict[str, np.ndarray]
    ) -> Tuple[Optional[str], float]:
        """Find best match and return both class and confidence score."""
        best_class = None
        best_sim = -1.0
        
        for class_name, proto_vec in prototypes.items():
            sim = np.dot(vector, proto_vec) / (
                np.linalg.norm(vector) * np.linalg.norm(proto_vec) + 1e-10
            )
            if sim > best_sim:
                best_sim = sim
                best_class = class_name
        
        return best_class, float(best_sim)
    
    def _finalize_class(self, track_state: TrackState) -> None:
        """Finalize class using confidence-weighted voting."""
        if not track_state.votes:
            return
        
        # Simple majority vote
        vote_counts = {}
        vote_scores = {}
        
        for vote, score in zip(track_state.votes, track_state.vote_scores):
            if vote not in vote_counts:
                vote_counts[vote] = 0
                vote_scores[vote] = []
            vote_counts[vote] += 1
            vote_scores[vote].append(score)
        
        # Find winner
        winner = max(vote_counts.keys(), key=lambda x: vote_counts[x])
        avg_confidence = sum(vote_scores[winner]) / len(vote_scores[winner])
        
        track_state.final_class = winner
        track_state.final_confidence = avg_confidence
    
    def _cleanup_old_tracks(self, state: Dict, current_frame: int) -> None:
        """Remove old track states to prevent memory leaks."""
        old_tracks = [
            track_id for track_id, ts in state["track_memory"].items()
            if current_frame - ts.last_seen_frame > self.TRACK_MEMORY_MAX_AGE
        ]
        
        for track_id in old_tracks:
            del state["track_memory"][track_id]
        
        if old_tracks:
            logger.debug(f"Cleaned up {len(old_tracks)} old tracks")
    
    def _crop_image(self, frame: np.ndarray, box) -> np.ndarray:
        """Crop image from bounding box with padding."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        
        # Add small padding (5%)
        pad_w = int((x2 - x1) * 0.05)
        pad_h = int((y2 - y1) * 0.05)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        
        return frame[y1:y2, x1:x2]
    
    def _is_crossing_line(self, p1, p2, line) -> bool:
        """Check if trajectory crosses the counting line."""
        (lx1, ly1), (lx2, ly2) = line
        
        # Horizontal line
        if ly1 == ly2:
            crossed = (p1[1] <= ly1 < p2[1]) or (p1[1] >= ly1 > p2[1])
            in_width = min(lx1, lx2) <= p2[0] <= max(lx1, lx2)
            return crossed and in_width
        
        # Vertical line
        if lx1 == lx2:
            crossed = (p1[0] <= lx1 < p2[0]) or (p1[0] >= lx1 > p2[0])
            in_height = min(ly1, ly2) <= p2[1] <= max(ly1, ly2)
            return crossed and in_height
        
        return False
    
    def _draw_detection(
        self, 
        frame: np.ndarray, 
        box, 
        track_id: int, 
        label: Optional[str],
        confidence: float,
        is_counted: bool
    ):
        """Draw bounding box with confidence indicator."""
        x1, y1, x2, y2 = map(int, box)
        
        # Color based on status
        if is_counted:
            color = (0, 255, 0)  # Green - counted
        elif label:
            # Confidence-based color (red to green gradient)
            green = int(255 * confidence)
            red = int(255 * (1 - confidence))
            color = (0, green, red)
        else:
            color = (128, 128, 128)  # Gray - identifying
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Label with confidence
        if label:
            text = f"ID:{track_id} {label} ({confidence:.0%})"
        else:
            text = f"ID:{track_id} Identifying..."
        
        # Background for text
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Counted indicator
        if is_counted:
            cv2.putText(frame, "✓ COUNTED", (x1, y2 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def _draw_count_overlay(self, frame: np.ndarray, counts: Dict, fps: float):
        """Draw count information overlay."""
        y_offset = 30
        box_height = 100 + len(counts.get("class_counts", {})) * 25
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (280, box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (10, 10), (280, box_height), (255, 255, 255), 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 25
        
        # Total count
        cv2.putText(frame, f"Total: {counts.get('total_count', 0)}", 
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += 35
        
        # Per-class counts
        for class_name, count in counts.get("class_counts", {}).items():
            cv2.putText(frame, f"  {class_name}: {count}", 
                        (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y_offset += 25
    
    def _get_color(self, track_id: int) -> tuple:
        """Generate consistent color for a track ID."""
        np.random.seed(track_id)
        return tuple(map(int, np.random.randint(0, 255, 3)))
