from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.api.v1.router import api_router
from app.api.v1.deps import init_ml_models, cleanup_ml_models
from app.db.session import init_db, close_db
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    # Startup
    logger.info("Starting Magic Vision API...")
    
    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized with pgvector extension")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    
    # Initialize ML models
    try:
        init_ml_models()
        logger.info("ML models loaded (YOLO + DINOv2)")
    except Exception as e:
        logger.error(f"ML model initialization failed: {e}")
        raise
    
    logger.info("Magic Vision API ready!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Magic Vision API...")
    cleanup_ml_models()
    await close_db()
    logger.info("Cleanup complete")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    ## Few-Shot Visual Object Counting System
    
    A real-time visual object counting system for industrial environments.
    
    ### Features:
    - **Object Library**: Register objects with 3-10 reference images
    - **Few-Shot Recognition**: No retraining needed for new objects
    - **Real-time Counting**: WebSocket-based video streaming
    - **Session Management**: Track counting history
    
    ### Workflow:
    1. Create objects in the library with reference images
    2. Create a counting session with target objects
    3. Start the session and connect via WebSocket
    4. Stream video frames for real-time counting
    5. Stop session to save results
    """,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "service": settings.APP_NAME
    }


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Magic Vision API",
        "docs": "/docs",
        "health": "/health"
    }