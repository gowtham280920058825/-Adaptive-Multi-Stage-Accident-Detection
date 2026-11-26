import sys
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from backend.detection.yolov11_amf import YOLO11AMFDetector
from backend.amave.engine import AMAVEEngine
from backend.detection.temporal_engine import TemporalEngine
from backend.detection.video_processor import VideoProcessor
from backend.config import CONFIDENCE_THRESHOLD, VERIFICATION_THRESHOLD, YOLO_AMF_WEIGHTS

detector = YOLO11AMFDetector(weights_path=YOLO_AMF_WEIGHTS, conf_thresh=CONFIDENCE_THRESHOLD)
amave = AMAVEEngine(verification_threshold=VERIFICATION_THRESHOLD)

def inspect_video(video_path):
    print(f"=== INSPECTING {video_path.name} ===")
    vp = VideoProcessor(video_path)
    temp_engine = TemporalEngine(window_size=10)
    frames_cache = []
    
    verified_count = 0
    total_frames = 0
    for frame_idx, timestamp, frame in vp.extract_frames(stride=2):
        total_frames += 1
        frames_cache.append(frame)
        dets = detector.detect_frame(frame)
        temp_engine.push_frame_detections(frame_idx, timestamp, dets)
        
        history_window = list(temp_engine.history)
        frame_seq_window = frames_cache[max(0, len(frames_cache) - 2):]
        
        res = amave.verify_event(
            current_detections=dets,
            temporal_history=history_window,
            frame_sequence=frame_seq_window
        )
        if res.get('verified', False):
            verified_count += 1
            print(f"Frame {frame_idx:3d}: VERIFIED! fused={res['fused_verification_score']:.4f}, det={res['detection_confidence']:.2f}, temp={res['temporal_persistence_score']:.2f}, mot={res['motion_consistency_score']:.2f}, int={res['interaction_score']:.2f}")
        elif dets:
            print(f"Frame {frame_idx:3d}: UNVERIFIED. fused={res['fused_verification_score']:.4f}, det={res['detection_confidence']:.2f}, temp={res['temporal_persistence_score']:.2f}, mot={res['motion_consistency_score']:.2f}, int={res['interaction_score']:.2f} | Reason: {res['reason'][:60]}")
    vp.release()
    print(f"Total frames: {total_frames}, Verified frames: {verified_count}\n")

if __name__ == "__main__":
    inspect_video(Path("dataset/evaluation_videos/non_accident/normal_traffic_clear.mp4"))
    inspect_video(Path("dataset/evaluation_videos/accident/nAXTthLfgtI_00.mp4"))
    inspect_video(Path("dataset/evaluation_videos/accident/__WFqm4i3vE_00.mp4"))
