import cv2
from pathlib import Path
import sys
import json

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from backend.detection.video_processor import VideoProcessor
from backend.detection.temporal_engine import TemporalEngine
from backend.detection.yolov11_amf import YOLO11AMFDetector
from backend.amave.engine import AMAVEEngine
from backend.config import YOLO_AMF_WEIGHTS, VERIFICATION_THRESHOLD, MIN_VERIFIED_FRAMES

detector = YOLO11AMFDetector(weights_path=YOLO_AMF_WEIGHTS, conf_thresh=0.20)
amave_engine = AMAVEEngine(verification_threshold=VERIFICATION_THRESHOLD)

def run_test(vpath, name, expected_gt):
    vp = VideoProcessor(vpath)
    temp_engine = TemporalEngine(window_size=10)
    frames_cache = []
    results = []
    
    for f_idx, ts, frame in vp.extract_frames(stride=2):
        frames_cache.append(frame)
        dets = detector.detect_frame(frame)
        temp_engine.push_frame_detections(f_idx, ts, dets)
        
        hist = list(temp_engine.history)
        seq = frames_cache[max(0, len(frames_cache) - 2):]
        
        res = amave_engine.verify_event(
            current_detections=dets,
            temporal_history=hist,
            frame_sequence=seq
        )
        if dets or res.get('verified', False):
            results.append((f_idx, ts, dets, res))
            
    vp.release()
    
    verified_frames = [r for r in results if r[3]['verified']]
    is_verified = (len(verified_frames) >= MIN_VERIFIED_FRAMES)
    final_pred = "ACCIDENT DETECTED" if is_verified else "NO ACCIDENT DETECTED"
    correct = (final_pred == expected_gt)
    
    peak_res = max([r[3] for r in results], key=lambda x: x.get('fused_verification_score', 0)) if results else {'fused_verification_score': 0}
    max_det_conf = max([max([d['confidence'] for d in r[2]], default=0) for r in results], default=0)
    
    print(f"\n==========================================")
    print(f"VIDEO: {name}")
    print(f"Ground Truth: {expected_gt}")
    print(f"YOLO Frames with detections: {len(results)}")
    print(f"Max YOLO Detection Confidence: {max_det_conf:.4f}")
    print(f"AMAVE Verified Frames: {len(verified_frames)}")
    print(f"Peak Fused Score: {peak_res.get('fused_verification_score', 0):.4f}")
    print(f"Final Decision: {final_pred}")
    print(f"Correct: {correct}")
    print(f"==========================================")
    return {
        "name": name,
        "gt": expected_gt,
        "final_pred": final_pred,
        "correct": correct,
        "verified_frames": len(verified_frames),
        "peak_fused_score": peak_res.get('fused_verification_score', 0)
    }

res1 = run_test(BASE_DIR / 'dataset/real_videos/__WFqm4i3vE_00.mp4', '__WFqm4i3vE_00.mp4 (Real Accident)', 'ACCIDENT DETECTED')
res2 = run_test(BASE_DIR / 'dataset/test_normal_traffic.mp4', 'test_normal_traffic.mp4 (Normal Highway)', 'NO ACCIDENT DETECTED')
