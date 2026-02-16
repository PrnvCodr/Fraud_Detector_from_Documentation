"""
DocFraudDetector — Central Configuration
All hyperparameters, paths, thresholds, and device settings.
"""

import os
import torch

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SAMPLE_IMAGES_DIR = os.path.join(DATA_DIR, "sample_images")
SYNTHETIC_DIR = os.path.join(DATA_DIR, "synthetic")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Create dirs if they don't exist
for d in [SAMPLE_IMAGES_DIR, SYNTHETIC_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# DEVICE
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# DOCUMENT DETECTION (YOLOv8)
# =============================================================================
YOLO_MODEL = "yolov8n.pt"  # Nano model for speed
YOLO_CONF_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.45
DOCUMENT_CLASSES = ["document"]

# =============================================================================
# PERSPECTIVE RECTIFICATION
# =============================================================================
RECTIFIED_WIDTH = 600
RECTIFIED_HEIGHT = 400
CANNY_LOW = 50
CANNY_HIGH = 150
CONTOUR_APPROX_EPSILON = 0.02  # Fraction of perimeter for approxPolyDP
GAUSSIAN_BLUR_KERNEL = (5, 5)

# =============================================================================
# TAMPER DETECTION
# =============================================================================
TAMPER_MODEL_NAME = "efficientnet_b0"
TAMPER_INPUT_SIZE = (224, 224)
TAMPER_THRESHOLD = 0.5  # Probability above this = tampered
TAMPER_NUM_CLASSES = 2
TAMPER_CLASS_NAMES = ["genuine", "tampered"]
TAMPER_CHECKPOINT = os.path.join(MODEL_DIR, "tamper_efficientnet_b0.pth")

# ELA (Error Level Analysis)
ELA_QUALITY = 90  # JPEG compression quality for ELA
ELA_SCALE = 10    # Amplification factor for difference

# =============================================================================
# OCR
# =============================================================================
OCR_LANGUAGES = ["en"]  # Add "hi" for Hindi support
OCR_GPU = torch.cuda.is_available()
OCR_CONFIDENCE_THRESHOLD = 0.3

# Document field patterns (regex)
FIELD_PATTERNS = {
    "name": r"(?:name|naam)\s*[:\-]?\s*([A-Za-z\s\.]+)",
    "dob": r"(?:dob|date\s*of\s*birth|d\.o\.b)\s*[:\-]?\s*([\d]{2}[\/\-][\d]{2}[\/\-][\d]{4})",
    "id_number": r"(?:no|number|id)\s*[:\-]?\s*([\dA-Z]{4,}[\s]?[\dA-Z]*)",
    "address": r"(?:address|addr)\s*[:\-]?\s*(.+?)(?:\n|$)",
    "gender": r"(?:gender|sex)\s*[:\-]?\s*(male|female|m|f)",
}

# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================
SYNTHETIC_NUM_GENUINE = 200
SYNTHETIC_NUM_TAMPERED = 200
SYNTHETIC_IMAGE_SIZE = (600, 400)

# Tamper types and probabilities
TAMPER_TYPES = {
    "text_replacement": 0.3,
    "font_mismatch": 0.2,
    "copy_paste": 0.2,
    "blur_injection": 0.15,
    "noise_injection": 0.15,
}

# =============================================================================
# TRAINING
# =============================================================================
TRAIN_BATCH_SIZE = 16
TRAIN_EPOCHS = 20
TRAIN_LR = 1e-4
TRAIN_WEIGHT_DECAY = 1e-5
TRAIN_VAL_SPLIT = 0.2
TRAIN_NUM_WORKERS = 0  # 0 for Windows compatibility
TRAIN_SCHEDULER_PATIENCE = 3

# =============================================================================
# API
# =============================================================================
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# =============================================================================
# DEMO
# =============================================================================
STREAMLIT_PORT = 8501
