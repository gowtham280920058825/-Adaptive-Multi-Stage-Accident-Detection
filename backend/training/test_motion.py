import cv2
import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from backend.detection.yolov11_amf import YOLO11AMFDetector
from backend.detection.video_processor import VideoProcessor
from backend.amave.temporal_verifier import TemporalVerifier
from backend.config import YOLO_AMF_WEIGHTS

detector = YOLO11AMFDetector(weights_path=YOLO_AMF_WEIGHTS, conf_thresh=0.20)
verifier = TemporalVerifier()

def test_vid(vpath, name):
    vp = VideoProcessor(vpath)
    frames_cache = []
    turb_records = []
    
    for f_idx, ts, frame in vp.extract_frames(stride=2):
        frames_cache.append(frame)
        dets = detector.detect_frame(frame)
        if len(frames_cache) >= 2 and dets:
            boxes = [d['box'] for d in dets]
            m_score = verifier.evaluate_motion_consistency(frames_cache[-2:], boxes)
            turb_records.append((f_idx, len(dets), max([d['confidence'] for d in dets]), m_score))
    vp.release()
    print(f"\n--- {name} --- (Frames with dets: {len(turb_records)})")
    for r in turb_records[:8]:
        print(f"  Frame {r[0]:3d}: dets={r[1]}, max_conf={r[2]:.3f}, motion_score={r[3]:.3f}")
    if len(turb_records) > 8:
        for r in turb_records[-4:]:
            print(f"  Frame {r[0]:3d}: dets={r[1]}, max_conf={r[2]:.3f}, motion_score={r[3]:.3f}")

test_vid(BASE_DIR / 'dataset/test_normal_traffic.mp4', 'Normal Traffic')
test_vid(BASE_DIR / 'dataset/real_videos/__WFqm4i3vE_00.mp4', 'Accident Video (__WFqm4i3vE_00)')
