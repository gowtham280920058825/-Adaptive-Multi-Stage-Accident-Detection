"""
Metrics Utility Module
Calculates model metrics (Precision, Recall, mAP50, mAP50-95, F1) and system metrics (False Positive Rate, Verification Accuracy, FPS).
"""

from typing import Dict, Any, List
import numpy as np

def compute_model_metrics(
    tp: int, fp: int, fn: int, tn: int = 0
) -> Dict[str, float]:
    """Calculate object detection / classification statistical metrics."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "accuracy": round(float(accuracy), 4),
        "false_positive_rate": round(float(fpr), 4),
    }

def compute_system_performance(
    total_frames: int, processing_time_seconds: float
) -> Dict[str, float]:
    """Calculate end-to-end processing frame rate and latency."""
    fps = total_frames / processing_time_seconds if processing_time_seconds > 0 else 0.0
    avg_latency_ms = (processing_time_seconds / total_frames) * 1000 if total_frames > 0 else 0.0

    return {
        "fps": round(float(fps), 2),
        "avg_latency_ms": round(float(avg_latency_ms), 2),
        "total_processing_time_sec": round(float(processing_time_seconds), 3),
    }
