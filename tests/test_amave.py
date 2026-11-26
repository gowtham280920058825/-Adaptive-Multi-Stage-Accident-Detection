"""
AMAVE Verification Module Unit Tests
"""

import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from backend.amave.engine import AMAVEEngine

def test_amave_empty_detections():
    engine = AMAVEEngine()
    result = engine.verify_event([], [])
    assert result["verified"] is False
    assert result["verification_score"] == 0.0

def test_amave_persistent_detections():
    engine = AMAVEEngine()
    dets = [{"box": [10, 10, 100, 100], "confidence": 0.88, "class_id": 0, "label": "Accident"}]
    history = [{"frame_idx": i, "has_candidate": True, "detections": dets, "max_confidence": 0.88} for i in range(10)]
    result = engine.verify_event(dets, history)
    assert result["verified"] is True
    assert result["verification_score"] > 0.65
