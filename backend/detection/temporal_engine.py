"""
Temporal Engine Module
Maintains sliding frame buffer and computes temporal motion vectors and persistent detection metrics.
"""

from collections import deque
import numpy as np
from typing import List, Dict, Any

class TemporalEngine:
    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def push_frame_detections(self, frame_idx: int, timestamp: float, detections: List[Dict[str, Any]]):
        """Pushes single frame candidate detections into temporal buffer."""
        self.history.append({
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "detections": detections,
            "has_candidate": len(detections) > 0,
            "max_confidence": max([d["confidence"] for d in detections], default=0.0)
        })

    def get_persistence_ratio(self) -> float:
        """Calculates fraction of recent frames containing candidate accident detections."""
        if not self.history:
            return 0.0
        candidate_count = sum(1 for item in self.history if item["has_candidate"])
        return candidate_count / len(self.history)

    def get_mean_confidence(self) -> float:
        """Calculates mean candidate confidence across buffer."""
        if not self.history:
            return 0.0
        confs = [item["max_confidence"] for item in self.history if item["has_candidate"]]
        return float(np.mean(confs)) if confs else 0.0

    def reset(self):
        self.history.clear()
