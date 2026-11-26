"""
Video Processor Module
Handles video frame decoding, stream processing, batch reading, and video annotation rendering.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Dict, Any, List

class VideoProcessor:
    def __init__(self, video_path: Path):
        self.video_path = video_path
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps

    def extract_frames(self, stride: int = 1) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Generator yielding (frame_index, timestamp_seconds, bgr_frame)."""
        frame_idx = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break
            if frame_idx % stride == 0:
                timestamp = frame_idx / self.fps
                yield frame_idx, timestamp, frame
            frame_idx += 1

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

def draw_annotations(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    amave_result: Dict[str, Any] = None,
    sasa_result: Dict[str, Any] = None,
    sede_result: Dict[str, Any] = None,
    is_video_verified: bool = False
) -> np.ndarray:
    """Renders visual overlays, bounding boxes, and status badges on the video frame."""
    annotated = frame.copy()
    h, w, _ = annotated.shape

    is_frame_verified = amave_result.get("verified", False) if amave_result else False
    severity = sasa_result.get("severity", sasa_result.get("severity_level", "N/A")) if sasa_result else "N/A"
    action = sede_result.get("recommended_action", "NO_ACTION") if sede_result else "NO_ACTION"
    # Shorten action for HUD display
    if "RECOMMENDATION:" in action:
        short_action = action.split("RECOMMENDATION:", 1)[1].strip().split("&")[0].strip()
    else:
        short_action = action

    # Color scheme: Red for verified accident, green for normal traffic
    is_alert = is_frame_verified or (is_video_verified and len(detections) > 0)
    status_color = (0, 0, 255) if is_alert else (0, 255, 0)
    badge_bg = (20, 20, 20)

    # 1. Top Status Banner
    cv2.rectangle(annotated, (0, 0), (w, 55), badge_bg, -1)
    if is_video_verified or is_frame_verified:
        status_str = f"STATUS: ACCIDENT DETECTED & VERIFIED [{severity}]"
    else:
        status_str = "STATUS: NORMAL TRAFFIC MONITORING"
    cv2.putText(annotated, status_str, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

    # 2. Draw Detections
    for det in detections:
        box = [int(c) for c in det["box"]]
        conf = det["confidence"]
        label = det.get("label", "Accident")

        cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), status_color, 2)
        lbl_text = f"{label}: {conf * 100:.1f}%"
        cv2.putText(annotated, lbl_text, (max(10, box[0]), max(20, box[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # 3. Bottom Summary Info
    score_val = amave_result.get("fused_verification_score", amave_result.get("verification_score", 0.0)) if amave_result else 0.0
    cv2.rectangle(annotated, (0, h - 38), (w, h), (15, 15, 15), -1)
    if is_video_verified:
        info_str = f"AMAVE Score: {score_val * 100:.1f}% | Severity: {severity} | Action: {short_action}"
        cv2.putText(annotated, info_str, (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
    else:
        info_str = f"AMAVE Score: {score_val * 100:.1f}% (Below Threshold) | System Decision: NO ACCIDENT DETECTED"
        cv2.putText(annotated, info_str, (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 2)

    return annotated
