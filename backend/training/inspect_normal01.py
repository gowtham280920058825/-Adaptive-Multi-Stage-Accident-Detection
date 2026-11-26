import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from backend.detection.yolov11_amf import YOLO11AMFDetector
from backend.detection.temporal_engine import TemporalEngine
from backend.detection.video_processor import VideoProcessor
from backend.config import CONFIDENCE_THRESHOLD, YOLO_AMF_WEIGHTS
from backend.training.test_all_10_videos import evaluate_amave_candidate

detector = YOLO11AMFDetector(weights_path=YOLO_AMF_WEIGHTS, conf_thresh=CONFIDENCE_THRESHOLD)

vp = VideoProcessor(Path("dataset/evaluation_videos/non_accident/normal_traffic_01.mp4"))
temp_engine = TemporalEngine(window_size=10)
frames_cache = []

for frame_idx, timestamp, frame in vp.extract_frames(stride=2):
    frames_cache.append(frame)
    dets = detector.detect_frame(frame)
    temp_engine.push_frame_detections(frame_idx, timestamp, dets)
    history = list(temp_engine.history)
    seq = frames_cache[max(0, len(frames_cache)-2):]
    res = evaluate_amave_candidate(dets, history, seq, frame.shape)
    if res["verified"]:
        print(f"Frame {frame_idx:3d}: VERIFIED! fused={res['fused']:.4f} det={res['s_det']:.2f} temp={res['s_temp']:.2f} mot={res['s_motion']:.2f} int={res['s_inter']:.2f}")
    elif dets:
        print(f"Frame {frame_idx:3d}: UNVERIFIED. fused={res['fused']:.4f} det={res['s_det']:.2f} temp={res['s_temp']:.2f} mot={res['s_motion']:.2f} int={res['s_inter']:.2f}")
vp.release()
