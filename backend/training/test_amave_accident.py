import cv2
from pathlib import Path
from ultralytics import YOLO
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from backend.detection.video_processor import VideoProcessor
from backend.detection.temporal_engine import TemporalEngine
from backend.detection.yolov11_amf import YOLO11AMFDetector
from backend.amave.engine import AMAVEEngine
from backend.config import YOLO_AMF_WEIGHTS, VERIFICATION_THRESHOLD

detector = YOLO11AMFDetector(weights_path=YOLO_AMF_WEIGHTS, conf_thresh=0.18)
amave_engine = AMAVEEngine(verification_threshold=VERIFICATION_THRESHOLD)
vp = VideoProcessor(BASE_DIR / 'dataset/real_videos/__WFqm4i3vE_00.mp4')
temp_engine = TemporalEngine(window_size=10)

frames_cache = []
results = []

for frame_idx, timestamp, frame in vp.extract_frames(stride=2):
    frames_cache.append(frame)
    dets = detector.detect_frame(frame)
    temp_engine.push_frame_detections(frame_idx, timestamp, dets)
    
    history_window = list(temp_engine.history)
    frame_seq_window = frames_cache[max(0, len(frames_cache) - 2):]
    
    res = amave_engine.verify_event(
        current_detections=dets,
        temporal_history=history_window,
        frame_sequence=frame_seq_window
    )
    if res.get('fused_verification_score', 0) > 0.20 or dets:
        results.append((frame_idx, timestamp, dets, res))
        print(f"Frame {frame_idx:3d} (t={timestamp:.2f}s): dets={len(dets)}, fused={res['fused_verification_score']:.4f}, det_conf={res['detection_confidence']:.4f}, pers={res['temporal_persistence_score']:.4f}, mot={res['motion_consistency_score']:.4f}, inter={res['interaction_score']:.4f}, verified={res['verified']}")

vp.release()
verified = [r for r in results if r[3]['verified']]
print(f"\nTOTAL EVALUATED ACCIDENT FRAMES WITH DETECTIONS: {len(results)}")
print(f"TOTAL VERIFIED WINDOWS: {len(verified)}")
