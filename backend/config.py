"""
System Configuration Module
Master configuration file for Adaptive Multi-Stage Framework for Vision-Based Road Accident Detection
"""

import os
import torch
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_RAW_DIR = BASE_DIR / "dataset"
DATASET_PREPARED_DIR = BASE_DIR / "dataset" / "prepared"
RESULTS_DIR = BASE_DIR / "results"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
MODELS_DIR = BASE_DIR / "backend" / "models"

# Ensure runtime directories exist
for folder in [
    DATASET_PREPARED_DIR,
    RESULTS_DIR / "detections",
    RESULTS_DIR / "annotated_videos",
    RESULTS_DIR / "reports" / "dataset_sample_verification",
    RESULTS_DIR / "metrics",
    RESULTS_DIR / "plots",
    RESULTS_DIR / "logs",
    EXPERIMENTS_DIR / "baseline",
    EXPERIMENTS_DIR / "base_paper",
    EXPERIMENTS_DIR / "proposed",
    MODELS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

# Hardware Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA_AVAILABLE = torch.cuda.is_available()
GPU_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "CPU (Fallback)"

# Model Weights Config
BASELINE_YOLO_WEIGHTS = MODELS_DIR / "yolov11_baseline.pt"
YOLO_AMF_WEIGHTS = MODELS_DIR / "yolov11_amf.pt"

# Detection & Verification Parameters
CONFIDENCE_THRESHOLD = 0.20  # Candidate proposal threshold
IOU_THRESHOLD = 0.40  # NMS IoU threshold
TEMPORAL_WINDOW_SIZE = 10  # frames in sliding buffer
MIN_PERSISTENCE_FRAMES = 2  # minimum consecutive candidate detections
MIN_VERIFIED_FRAMES = 2  # minimum verified frames for event-level ACCIDENT DETECTED
VERIFICATION_THRESHOLD = 0.45  # AMAVE fused score cut-off

# AMAVE Confidence Fusion Weights
AMAVE_WEIGHTS = {
    "det": 0.30,
    "temp": 0.20,
    "motion": 0.30,
    "inter": 0.20
}

# SASA Severity Thresholds
SEVERITY_WEIGHTS = {
    "vehicle_count": 0.3,
    "impact_speed_proxy": 0.3,
    "collision_type": 0.2,
    "rollover": 0.2,
}

# SEDE Emergency Action Level Matrix
SEDE_POLICY = {
    "LOW": {
        "level": 1,
        "action": "RECORD_EVENT",
        "description": "Log event for traffic monitoring.",
    },
    "MODERATE": {
        "level": 2,
        "action": "NOTIFY_CONTROL_ROOM",
        "description": "Alert regional traffic operator control room.",
    },
    "HIGH": {
        "level": 3,
        "action": "DISPATCH_EMERGENCY_SERVICES",
        "description": "Request immediate emergency medical & police response.",
    },
    "CRITICAL": {
        "level": 4,
        "action": "MAXIMUM_PRIORITY_DISPATCH",
        "description": "Priority emergency dispatch, route clearance, & medical alert.",
    },
}
