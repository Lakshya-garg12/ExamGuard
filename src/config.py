"""
ExamGuard - AI-Based Exam Environment Monitoring System
Configuration & Master Settings
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
VIDEOS_DIR = BASE_DIR / "videos"
EVIDENCE_DIR = DATA_DIR / "alerts_evidence"
RUNS_DIR = BASE_DIR / "runs"
DB_PATH = BASE_DIR / "examguard.db"

# Ensure runtime directories exist
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Default Model Paths
DEFAULT_CUSTOM_MODEL = MODELS_DIR / "trained" / "best.pt"
DEFAULT_PRETRAINED_MODEL = MODELS_DIR / "yolo11s.pt"  # Upgraded to high-accuracy YOLO11s model
DEFAULT_NANO_MODEL = MODELS_DIR / "yolo11n.pt"

# Master Object Classes & Severity Configuration
# AI acts as an assistant - it classifies detections and assigns alert severities.
CLASS_CONFIG = {
    # Prohibited & Suspicious Items (Trigger alerts for invigilator review)
    "cell phone": {
        "severity": "HIGH",
        "label": "Mobile Phone",
        "description": "Unauthorized electronic communication device",
        "color": (0, 0, 255),  # Red (BGR)
    },
    "mobile_phone": {
        "severity": "HIGH",
        "label": "Mobile Phone",
        "description": "Unauthorized electronic communication device",
        "color": (0, 0, 255),
    },
    "laptop": {
        "severity": "HIGH",
        "label": "Laptop / Tablet",
        "description": "Unauthorized computing device",
        "color": (0, 0, 255),
    },
    "headphones": {
        "severity": "HIGH",
        "label": "Headphones / Earphones",
        "description": "Unauthorized audio device",
        "color": (0, 0, 255),
    },
    "book": {
        "severity": "MEDIUM",
        "label": "Book / Notes",
        "description": "Potentially unauthorized reading material",
        "color": (0, 140, 255),  # Orange (BGR)
    },
    "paper": {
        "severity": "MEDIUM",
        "label": "Loose Paper / Sheet",
        "description": "Possible cheat sheet or unauthorized paper",
        "color": (0, 215, 255),  # Amber (BGR)
    },
    
    # Permitted Examination Items (Logged as normal/permitted activity)
    "pencil": {
        "severity": "INFO",
        "label": "Pencil",
        "description": "Permitted writing stationery",
        "color": (0, 255, 0),  # Green (BGR)
    },
    "pen": {
        "severity": "INFO",
        "label": "Pen",
        "description": "Permitted writing stationery",
        "color": (0, 255, 0),
    },
    "eraser": {
        "severity": "INFO",
        "label": "Eraser",
        "description": "Permitted stationery item",
        "color": (255, 255, 0),  # Cyan
    },
    "sharpener": {
        "severity": "INFO",
        "label": "Sharpener",
        "description": "Permitted stationery item",
        "color": (255, 255, 0),
    },
    "ruler": {
        "severity": "INFO",
        "label": "Ruler",
        "description": "Permitted stationery item",
        "color": (255, 200, 0),
    },
    "calculator": {
        "severity": "MEDIUM",
        "label": "Calculator",
        "description": "Calculator (Verify if allowed in this exam)",
        "color": (0, 165, 255),
    },
}

# Detection & Tracking Thresholds
DEFAULT_CONF_THRESHOLD = 0.30
DEFAULT_IOU_THRESHOLD = 0.45
ALERT_COOLDOWN_SECONDS = 3.0  # Avoid duplicate alerts for the same track within 3 seconds
MIN_DETECTION_FRAMES = 3      # Consecutive frames an object must appear to confirm an alert

# Camera Settings
DEFAULT_CAMERA_ID = "ExamHall-Cam-01"
DEFAULT_VIDEO_SOURCE = 0      # 0 for default webcam, or path to video file
STREAM_FPS = 25
STREAM_WIDTH = 1280
STREAM_HEIGHT = 720

