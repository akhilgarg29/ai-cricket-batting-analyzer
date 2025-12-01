"""
Configuration settings for AI Cricket Batting Analyzer
"""
import os
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
FRONTEND_DIR = BASE_DIR / "frontend"

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# Processing Settings
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "2"))  # Process every Nth frame
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))

# Model Paths
YOLO_MODEL_PATH = "yolov8n.pt"

# MediaPipe Settings
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5
POSE_MODEL_COMPLEXITY = 2  # 0, 1, or 2 (higher = more accurate but slower)

# Analysis Settings
MIN_CONFIDENCE_THRESHOLD = 0.6
MAX_ISSUES_TO_DISPLAY = 8

# Server Settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
RELOAD = os.getenv("RELOAD", "True").lower() == "true"

# Cleanup Settings
AUTO_CLEANUP_HOURS = int(os.getenv("AUTO_CLEANUP_HOURS", "24"))  # Delete old files after N hours
