"""
Configuration and Global Constants for
AI-Based Mechanical Lab & Industrial Worker Safety Monitoring System.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "safety_records.db"
RECORDS_DIR = BASE_DIR / "records"
CSV_PATH = RECORDS_DIR / "safety_records.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best.pt"
DATASET_DIR = BASE_DIR / "dataset"
TRAINING_DIR = BASE_DIR / "training"

# Ensure runtime directories exist
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
RECORDS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Application Metadata
APP_TITLE = "AI INDUSTRIAL SAFETY MONITORING SYSTEM"
APP_SUBTITLE = "Mechanical Laboratory & Industrial PPE Compliance Verification"
APP_VERSION = "2.0.0"

# Target Detection Classes
# 0: Person, 1: Safety Helmet, 2: Safety Cap, 3: Safety Shoes, 4: No Helmet, 5: No Safety Shoes
CLASSES = {
    0: "person",
    1: "safety_helmet",
    2: "safety_cap",
    3: "safety_shoes",
    4: "no_helmet",
    5: "no_safety_shoes"
}

# Supported Safety Status
class SafetyStatus:
    SAFE = "SAFE"
    NOT_SAFE = "NOT_SAFE"
    UNKNOWN = "UNKNOWN"

# Color Codes (Hex and BGR for OpenCV) - Vibrant Pink, Green, Yellow Palette
STATUS_COLORS = {
    SafetyStatus.SAFE: "#10B981",      # Radiant Mint/Emerald Green
    SafetyStatus.NOT_SAFE: "#F43F5E",  # Cyber Neon Pink/Rose
    SafetyStatus.UNKNOWN: "#FACC15"   # Electric Yellow / Gold
}

# Vibrant UI Theme Accents
THEME_WHITE = "#FFFFFF"
THEME_PINK = "#EC4899"
THEME_PINK_LIGHT = "#F43F5E"
THEME_GREEN = "#10B981"
THEME_GREEN_NEON = "#00F5A0"
THEME_YELLOW = "#FACC15"
THEME_YELLOW_GOLD = "#FFD600"

STATUS_BGR = {
    SafetyStatus.SAFE: (160, 245, 0),      # Neon Mint Green BGR
    SafetyStatus.NOT_SAFE: (94, 63, 244),   # Cyber Pink/Rose BGR
    SafetyStatus.UNKNOWN: (0, 214, 255)    # Electric Yellow/Gold BGR
}

# Detection & Tracking Defaults
DEFAULT_CONF_THRESHOLD = 0.25   # Lowered from 0.40 to ensure laptop webcams pick up humans easily
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_IMGSZ = 480
DEFAULT_CAMERA_INDEX = 0
DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480
FRAME_EDGE_MARGIN = 15          # Boundary pixels to detect head/feet truncation
HYSTERESIS_WINDOW = 4           # Responsive consecutive frames for temporal stabilization
RECORD_COOLDOWN_SECONDS = 15.0  # Session deduplication cooldown
ENABLE_FACE_SCANNER = True      # High-speed OpenCV facial detector & scanning reticle


# Workplace Safety Policy Options
ALLOW_SAFETY_CAP = True         # Allow approved safety cap as alternative to hard helmet in lab
STRICT_HELMET_ONLY = False      # Require hard helmet specifically for high-risk zones

