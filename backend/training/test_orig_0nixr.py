import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from backend.detection.yolov11_amf import YOLO11AMFDetector
from backend.amave.engine import AMAVEEngine
from backend.detection.temporal_engine import TemporalEngine
from backend.detection.video_processor import VideoProcessor
from backend.config import CONFIDENCE_THRESHOLD, VERIFICATION_THRESHOLD, YOLO_AMF_WEIGHTS

detector = YOLO11AMFDetector(weights_path=YOLO_AMF_WEIGHTS, conf_thresh=CONFIDENCE_THRESHOLD)
amave = AMAVEEngine(verification_threshold=VERIFICATION_THRESHOLD)

vp = VideoProcessor(Path("dataset/evaluation_videos/accident/0NixrF6Mt00_0_00.mp4"))
temp_engine = TemporalEngine(window_size=10)
frames_cache = []

for frame_idx, timestamp, frame in vp.extract_frames(stride=2):
    frames_cache.append(frame)
    dets = detector.detect_frame(frame)
    temp_engine.push_frame_detections(frame_idx, timestamp, dets)
    history = list(temp_engine.history)
    seq = frames_cache[max(0, len(frames_cache)-2):]
    res = amave.verify_event(dets, history, seq)
    if res["verified"]:
        print(f"Frame {frame_idx:3d}: VERIFIED! fused={res['fused_verification_score']:.4f} det={res['detection_confidence']:.2f} temp={res['temporal_persistence_score']:.2f} mot={res['motion_consistency_score']:.2f} int={res['interaction_score']:.2f}")
vp.release()
