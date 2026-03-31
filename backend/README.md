# Few-Shot Visual Object Counting System

A production-ready computer vision system for real-time object counting in industrial environments. The system supports arbitrary object categories using few-shot learning, requiring only 3-10 reference images per class without model retraining.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Technical Pipeline](#technical-pipeline)
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [WebSocket Protocol](#websocket-protocol)
- [Operational Workflow](#operational-workflow)
- [Deployment](#deployment)
- [Performance Considerations](#performance-considerations)
- [Limitations](#limitations)

## Overview

This system addresses the challenge of object counting in manufacturing environments where:

- Manual counting is error-prone and inefficient
- Weight-based estimation is impractical for large or irregular objects
- Frequent introduction of new object types renders static models obsolete
- Large labeled datasets are unavailable for training

The solution employs a few-shot learning approach that enables rapid adaptation to new object categories through minimal reference imagery, eliminating the need for extensive data collection or model retraining.

## System Architecture

The system follows Clean Architecture principles with clear separation of concerns:

```
                                    +------------------+
                                    |   Frontend/UI    |
                                    +--------+---------+
                                             |
                              WebSocket / REST API
                                             |
+------------------------------------------------------------------------------------+
|                                    Backend Service                                  |
|                                                                                     |
|  +----------------+    +------------------+    +--------------------------------+   |
|  |   API Layer    |    |  Service Layer   |    |         ML Pipeline            |   |
|  |                |    |                  |    |                                |   |
|  | - REST Routes  |--->| - ObjectService  |--->| YOLOv8 -> DINOv2 -> ByteTrack  |   |
|  | - WebSocket    |    | - SessionService |    |    |         |          |      |   |
|  | - Validation   |    | - CountingService|    | Detect   Embed    Track/Count  |   |
|  +----------------+    +------------------+    +--------------------------------+   |
|           |                     |                                                   |
|           v                     v                                                   |
|  +----------------+    +------------------+                                         |
|  | Repository     |    |   Storage        |                                         |
|  |                |    |                  |                                         |
|  | - PostgreSQL   |    | - MinIO (S3)     |                                         |
|  | - pgvector     |    | - Image Storage  |                                         |
|  +----------------+    +------------------+                                         |
+------------------------------------------------------------------------------------+
```

### Component Responsibilities

| Layer | Component | Responsibility |
|-------|-----------|----------------|
| API | REST Endpoints | CRUD operations for objects and sessions |
| API | WebSocket Handler | Real-time video frame processing |
| Service | ObjectService | Object registration, embedding computation |
| Service | SessionService | Session lifecycle management, count tracking |
| Service | CountingService | ML pipeline orchestration, tracking logic |
| Repository | ObjectRepository | Object persistence, vector similarity search |
| Repository | SessionRepository | Session persistence, history management |
| ML | YOLOv8 | Class-agnostic object detection |
| ML | DINOv2 | Feature extraction for few-shot matching |
| ML | ByteTrack | Multi-object tracking across frames |
| Storage | MinIO | Reference image storage |
| Database | PostgreSQL + pgvector | Metadata and embedding storage |

## Technical Pipeline

The counting pipeline processes video frames through four stages:

```
Frame Input
     |
     v
+--------------------+
| 1. DETECTION       |  YOLOv8 (class-agnostic)
| - Localize objects |  Output: Bounding boxes
+--------------------+
     |
     v
+--------------------+
| 2. EMBEDDING       |  DINOv2 ViT-S/14
| - Extract features |  Output: 384-dim vectors
| - Batch processing |  
+--------------------+
     |
     v
+--------------------+
| 3. MATCHING        |  Cosine Similarity
| - Compare to refs  |  Confidence-weighted voting
| - Classify objects |  Minimum threshold: 65%
+--------------------+
     |
     v
+--------------------+
| 4. TRACKING        |  ByteTrack
| - Assign track IDs |  Line-crossing detection
| - Count crossings  |  Unique ID management
+--------------------+
     |
     v
Count Result
```

### Key Optimizations

1. **Batch Encoding**: Multiple crops processed in single forward pass
2. **Confidence Scoring**: Weighted voting with early exit for high-confidence matches
3. **Memory Management**: Automatic cleanup of stale track states
4. **Frame Skipping**: Configurable processing frequency for performance tuning

## Directory Structure

```
backend/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/
│   │       │   ├── objects.py      # Object Library endpoints
│   │       │   ├── sessions.py     # Session Management endpoints
│   │       │   └── stream.py       # WebSocket streaming
│   │       ├── deps.py             # Dependency injection
│   │       └── router.py           # Route aggregation
│   ├── core/
│   │   ├── config.py               # Application settings
│   │   └── exceptions.py           # Custom exception handlers
│   ├── db/
│   │   ├── models/
│   │   │   ├── object.py           # Object and ObjectImage models
│   │   │   └── session.py          # CountingSession model
│   │   └── session.py              # Database session management
│   ├── ml/
│   │   ├── detector.py             # YOLOv8 wrapper
│   │   ├── encoder.py              # DINOv2 encoder
│   │   └── tracker.py              # ByteTrack implementation
│   ├── repositories/
│   │   ├── object_repository.py    # Object data access
│   │   └── session_repository.py   # Session data access
│   ├── schemas/
│   │   ├── object.py               # Object request/response schemas
│   │   ├── session.py              # Session schemas
│   │   └── stream.py               # WebSocket message schemas
│   ├── services/
│   │   ├── counting_service.py     # Counting pipeline orchestration
│   │   ├── object_service.py       # Object business logic
│   │   └── session_service.py      # Session business logic
│   ├── storage/
│   │   └── minio_client.py         # MinIO client wrapper
│   └── main.py                     # Application entry point
├── migrations/
│   ├── versions/
│   │   └── 001_initial.py          # Initial schema migration
│   └── env.py                      # Alembic configuration
├── models/                         # ML model weights directory
├── alembic.ini                     # Alembic settings
├── Dockerfile                      # Container build specification
├── requirements.txt                # Python dependencies
└── README.md                       # This document
```

## Prerequisites

### System Requirements

- Python 3.10 or higher
- CUDA-compatible GPU (recommended for production)
- Docker and Docker Compose (for containerized deployment)

### External Services

- PostgreSQL 15+ with pgvector extension
- MinIO or S3-compatible object storage

### ML Model Files

The following model files must be present in the `models/` directory:

| File | Description | Source |
|------|-------------|--------|
| `yolov8n.pt` | YOLOv8 Nano detection model | [Ultralytics](https://docs.ultralytics.com/) |
| `dinov2_vits14.pth` | DINOv2 ViT-S/14 encoder | [Meta Research](https://github.com/facebookresearch/dinov2) |

## Installation

### Local Development

```bash
# Clone the repository
git clone <repository-url>
cd magic-vision

# Start infrastructure services
docker-compose up -d postgres minio

# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\Activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run database migrations
alembic upgrade head

# Start the development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation

After starting the server, access the interactive API documentation:

- OpenAPI (Swagger UI): http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Configuration

Configuration is managed through environment variables. All settings have sensible defaults for development.

### Database Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_HOST` | `localhost` | PostgreSQL server hostname |
| `POSTGRES_PORT` | `5432` | PostgreSQL server port |
| `POSTGRES_USER` | `magicvision` | Database username |
| `POSTGRES_PASSWORD` | `!magicvision1` | Database password |
| `POSTGRES_DB` | `magicvision` | Database name |

### Object Storage Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MINIO_HOST` | `localhost` | MinIO server hostname |
| `MINIO_PORT` | `9000` | MinIO server port |
| `MINIO_ROOT_USER` | `minioadmin` | MinIO access key |
| `MINIO_ROOT_PASSWORD` | `minioadmin` | MinIO secret key |
| `MINIO_BUCKET_NAME` | `magicvision` | Storage bucket name |

### ML Pipeline Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_MODEL_PATH` | `models/yolov8n.pt` | Path to YOLO model weights |
| `DINO_MODEL_SIZE` | `vits14` | DINOv2 model variant |
| `EMBEDDING_DIM` | `384` | Feature vector dimension |
| `DETECTION_CONFIDENCE` | `0.25` | Minimum detection confidence |
| `SIMILARITY_THRESHOLD` | `0.7` | Minimum similarity for matching |
| `MIN_MATCH_CONFIDENCE` | `0.65` | Minimum confidence to accept match |
| `HIGH_CONFIDENCE_THRESHOLD` | `0.85` | Threshold for early voting exit |
| `VOTE_COUNT` | `3` | Votes required for classification |
| `FRAME_SKIP` | `5` | Process every N frames |
| `MAX_TRACK_AGE` | `30` | Frames before track is lost |
| `TRACK_MEMORY_MAX_AGE` | `150` | Frames before track memory cleanup |

## API Reference

### Object Library

Manage the reference object database for few-shot recognition.

#### Create Object

```http
POST /api/v1/objects
Content-Type: multipart/form-data
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | Yes | Unique object identifier |
| `description` | string | No | Object description |
| `files` | file[] | Yes | Reference images (3-10 recommended) |

**Response**: `201 Created`
```json
{
  "id": "uuid",
  "name": "Screw_M4",
  "description": "M4 Phillips head screw",
  "image_count": 5,
  "is_active": true,
  "created_at": "2024-01-01T00:00:00Z"
}
```

#### List Objects

```http
GET /api/v1/objects?active_only=true&limit=50&offset=0
```

#### Get Object Details

```http
GET /api/v1/objects/{object_id}
```

#### Update Object

```http
PUT /api/v1/objects/{object_id}
Content-Type: application/json

{
  "name": "Updated Name",
  "description": "Updated description"
}
```

#### Add Reference Images

```http
POST /api/v1/objects/{object_id}/images
Content-Type: multipart/form-data
```

#### Delete Object

```http
DELETE /api/v1/objects/{object_id}?hard=false
```

Soft delete (default) marks the object as inactive. Hard delete permanently removes the object and associated images.

### Session Management

Control counting sessions and access historical data.

#### Create Session

```http
POST /api/v1/sessions
Content-Type: application/json

{
  "name": "Batch #001",
  "description": "Quality inspection batch",
  "target_object_ids": ["uuid-1", "uuid-2"],
  "camera_source": "0"
}
```

**Response**: `201 Created`
```json
{
  "id": "uuid",
  "name": "Batch #001",
  "description": "Quality inspection batch",
  "target_object_ids": ["uuid-1", "uuid-2"],
  "target_object_names": {"uuid-1": "Screw_M4", "uuid-2": "Bolt_M6"},
  "class_counts": {},
  "total_count": 0,
  "status": "created",
  "camera_source": "0",
  "start_time": null,
  "end_time": null,
  "created_at": "2024-01-01T00:00:00Z"
}
```

#### Start Session

```http
POST /api/v1/sessions/{session_id}/start
```

Initializes the counting pipeline and begins accepting video frames.

#### Stop Session

```http
POST /api/v1/sessions/{session_id}/stop
```

Finalizes counts and persists results to database.

**Response**:
```json
{
  "id": "uuid",
  "name": "Batch #001",
  "status": "completed",
  "class_counts": {"Screw_M4": 45, "Bolt_M6": 32},
  "total_count": 77,
  "start_time": "2024-01-01T10:00:00Z",
  "end_time": "2024-01-01T10:15:00Z"
}
```

#### Get Live Counts

```http
GET /api/v1/sessions/{session_id}/counts
```

Returns current counts for an active session.

#### List Sessions

```http
GET /api/v1/sessions?status=completed&limit=50&offset=0
```

Status filter options: `created`, `running`, `completed`, `cancelled`

#### Cancel Session

```http
POST /api/v1/sessions/{session_id}/cancel
```

Terminates session without saving results.

#### Delete Session

```http
DELETE /api/v1/sessions/{session_id}
```

Only available for non-running sessions.

## WebSocket Protocol

Real-time video streaming uses WebSocket connections for bidirectional communication.

### Connection

```
WebSocket: ws://localhost:8000/api/v1/stream/{session_id}
```

The session must be in `running` state prior to connection.

### Client to Server Messages

#### Video Frame

```json
{
  "type": "frame",
  "data": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

The `data` field accepts base64-encoded JPEG or PNG images with optional data URI prefix.

#### Counting Line Configuration

```json
{
  "type": "line_config",
  "start_point": [100, 300],
  "end_point": [500, 300]
}
```

Defines the line that objects must cross to be counted. Coordinates are in pixels relative to frame dimensions.

#### Session Control

```json
{
  "type": "stop"
}
```

### Server to Client Messages

#### Processing Result

```json
{
  "type": "result",
  "frame": "/9j/4AAQ...",
  "detections": 5,
  "class_counts": {"Screw_M4": 3, "Bolt_M6": 2},
  "total_count": 5,
  "fps": 15.2,
  "timestamp": 1704067200.123
}
```

The `frame` field contains base64-encoded annotated image with bounding boxes and count overlay.

#### Status Update

```json
{
  "type": "status",
  "status": "ready",
  "session_id": "uuid",
  "message": "Counting pipeline initialized"
}
```

#### Error

```json
{
  "type": "error",
  "message": "Session not found"
}
```

## Operational Workflow

### Phase 1: Object Registration

Register each object type with 3-10 reference images captured from various angles.

```bash
curl -X POST "http://localhost:8000/api/v1/objects" \
  -F "name=Screw_M4" \
  -F "description=M4 Phillips head screw, zinc-plated" \
  -F "files=@reference_1.jpg" \
  -F "files=@reference_2.jpg" \
  -F "files=@reference_3.jpg" \
  -F "files=@reference_4.jpg" \
  -F "files=@reference_5.jpg"
```

The system computes prototype embeddings from the reference images.

### Phase 2: Session Creation

Create a counting session specifying which objects to track.

```bash
curl -X POST "http://localhost:8000/api/v1/sessions" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Production Batch 2024-001",
    "description": "Final inspection before shipment",
    "target_object_ids": ["<screw-uuid>", "<bolt-uuid>"]
  }'
```

### Phase 3: Active Counting

Start the session and stream video frames through WebSocket.

```javascript
// Start session via REST
await fetch(`/api/v1/sessions/${sessionId}/start`, { method: 'POST' });

// Connect WebSocket
const ws = new WebSocket(`ws://localhost:8000/api/v1/stream/${sessionId}`);

// Configure counting line
ws.send(JSON.stringify({
  type: 'line_config',
  start_point: [0, 360],
  end_point: [640, 360]
}));

// Stream frames from camera
const video = document.querySelector('video');
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');

setInterval(() => {
  ctx.drawImage(video, 0, 0);
  ws.send(JSON.stringify({
    type: 'frame',
    data: canvas.toDataURL('image/jpeg', 0.8)
  }));
}, 100);

// Handle results
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'result') {
    updateDisplay(data.class_counts, data.total_count);
  }
};
```

### Phase 4: Session Completion

Stop the session to finalize and persist counts.

```bash
curl -X POST "http://localhost:8000/api/v1/sessions/<session-id>/stop"
```

### Phase 5: Historical Review

Query completed sessions for reporting and analysis.

```bash
curl "http://localhost:8000/api/v1/sessions?status=completed"
```

## Deployment

### Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down
```

### Container Build

```bash
# Build backend image
docker build -t magic-vision-backend:latest ./backend

# Run with external services
docker run -d \
  --name magic-vision-backend \
  -p 8000:8000 \
  -e POSTGRES_HOST=db.example.com \
  -e MINIO_HOST=storage.example.com \
  -v ./models:/app/models \
  magic-vision-backend:latest
```

### Production Considerations

1. **SSL/TLS**: Deploy behind a reverse proxy (nginx, Traefik) with TLS termination
2. **Scaling**: The application is stateless; horizontal scaling requires shared Redis for session state
3. **Monitoring**: Integrate with Prometheus/Grafana for metrics collection
4. **Logging**: Configure structured logging output for log aggregation systems

## Performance Considerations

### Recommended Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| GPU | - | NVIDIA RTX 3060+ |
| Storage | SSD 100GB | NVMe SSD 500GB |

### Throughput Benchmarks

| Configuration | Typical FPS | Latency |
|---------------|-------------|---------|
| CPU only | 5-8 | 120-200ms |
| GPU (RTX 3060) | 15-25 | 40-70ms |
| GPU (RTX 4090) | 30-45 | 20-35ms |

### Optimization Strategies

1. Increase `FRAME_SKIP` for higher effective throughput at reduced granularity
2. Reduce input resolution for faster processing
3. Enable `BATCH_ENCODING` for multiple simultaneous detections
4. Use GPU acceleration for both detection and encoding

## Limitations

### Technical Constraints

- **Intra-class Similarity**: Objects with high visual similarity may cause classification errors
- **Severe Occlusion**: Heavily occluded objects may not be detected or tracked correctly
- **ID Switching**: Rapid movement or crowded scenes may cause track ID reassignment
- **Lighting Sensitivity**: Significant lighting variations require diverse reference images

### Operational Boundaries

- Recommended maximum of 50 simultaneous tracked objects per frame
- Reference images should be captured under similar lighting conditions to operational environment
- Counting line must be positioned where objects pass sequentially (not in parallel)

---

For additional support or feature requests, please open an issue in the project repository.
