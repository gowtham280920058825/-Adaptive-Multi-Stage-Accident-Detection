"""
Base YOLO11 Accident Detector
Loads trained/pretrained YOLO11 model weights, runs frame-level detection,
and extracts candidate accident bounding boxes with confidence scores.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Any

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from backend.config import BASELINE_YOLO_WEIGHTS, DEVICE, CONFIDENCE_THRESHOLD, IOU_THRESHOLD
from backend.utils.logging_utils import get_logger

logger = get_logger("BaseYOLO11Detector")

class BaseYOLO11Detector:
    def __init__(self, weights_path: Path = BASELINE_YOLO_WEIGHTS, conf_thresh: float = CONFIDENCE_THRESHOLD, iou_thresh: float = IOU_THRESHOLD):
        self.weights_path = weights_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = DEVICE
        self.model = None

        self._load_model()

    def _load_model(self):
        """Load YOLO11 model weights. Falls back to yolov8n/yolov11n standard weights if custom weights missing."""
        try:
            if self.weights_path.exists():
                logger.info(f"Loading custom YOLO11 weights from {self.weights_path} on {self.device}")
                self.model = YOLO(str(self.weights_path))
            else:
                logger.info(f"Custom weights not found at {self.weights_path}. Loading standard base model on {self.device}")
                self.model = YOLO("yolov8n.pt")  # Baseline fallback
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
            self.model = YOLO("yolov8n.pt")

    def detect_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs object/accident detection on a single frame.
        Returns candidate bounding boxes: [{"box": [x1, y1, x2, y2], "confidence": float, "class_id": int, "label": str}]
        """
        if self.model is None:
            return []

        results = self.model(frame, conf=self.conf_thresh, iou=self.iou_thresh, device=self.device, verbose=False)
        candidates = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                label = self.model.names.get(cls_id, "object")

                candidates.append({
                    "box": [round(c, 2) for c in xyxy],
                    "confidence": round(conf, 4),
                    "class_id": cls_id,
                    "label": label
                })

        return candidates
