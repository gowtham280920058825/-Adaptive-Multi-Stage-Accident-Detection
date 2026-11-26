import sys
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from backend.detection.yolov11_amf import YOLO11AMFDetector
from backend.detection.video_processor import VideoProcessor
from backend.config import CONFIDENCE_THRESHOLD, YOLO_AMF_WEIGHTS

detector = YOLO11AMFDetector(weights_path=YOLO_AMF_WEIGHTS, conf_thresh=CONFIDENCE_THRESHOLD)

def analyze_motion(video_path, max_frames=120):
    print(f"=== MOTION ANALYSIS: {video_path.name} ===")
    vp = VideoProcessor(video_path)
    frames = []
    for frame_idx, timestamp, frame in vp.extract_frames(stride=2):
        frames.append(frame)
        dets = detector.detect_frame(frame)
        if len(frames) >= 2 and dets:
            f1, f2 = frames[-2], frames[-1]
            h_orig, w_orig = f1.shape[:2]
            h_proc, w_proc = 240, 320
            f1_s = cv2.resize(f1, (w_proc, h_proc))
            f2_s = cv2.resize(f2, (w_proc, h_proc))
            g1 = cv2.cvtColor(f1_s, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(f2_s, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 2, 11, 2, 5, 1.1, 0)
            u, v = flow[..., 0], flow[..., 1]
            scene_u, scene_v = float(np.median(u)), float(np.median(v))
            scene_mag = np.sqrt(scene_u**2 + scene_v**2)
            
            # Target box
            for d in dets:
                box = d['box']
                sx, sy = w_proc / max(1.0, float(w_orig)), h_proc / max(1.0, float(h_orig))
                x1, y1 = max(0, int(box[0] * sx)), max(0, int(box[1] * sy))
                x2, y2 = min(w_proc, int(box[2] * sx)), min(h_proc, int(box[3] * sy))
                if x2 > x1 and y2 > y1:
                    raw_u = u[y1:y2, x1:x2]
                    raw_v = v[y1:y2, x1:x2]
                    raw_mag = float(np.mean(np.sqrt(raw_u**2 + raw_v**2)))
                    
                    b_u = raw_u - scene_u
                    b_v = raw_v - scene_v
                    b_mag = np.sqrt(b_u**2 + b_v**2)
                    mean_mag = float(np.mean(b_mag))
                    std_mag = float(np.std(b_mag))
                    angles = np.arctan2(b_v, b_u)
                    angle_var = float(np.var(angles))
                    print(f"F{frame_idx:3d} conf={d['confidence']:.2f}: scene_mag={scene_mag:.2f} raw_mag={raw_mag:.2f} mean_rel_mag={mean_mag:.2f} std_mag={std_mag:.2f} angle_var={angle_var:.2f}")
        if frame_idx > max_frames:
            break
    vp.release()

if __name__ == "__main__":
    analyze_motion(Path("dataset/evaluation_videos/non_accident/normal_traffic_clear.mp4"), 40)
    analyze_motion(Path("dataset/evaluation_videos/accident/nAXTthLfgtI_00.mp4"), 120)
    analyze_motion(Path("dataset/evaluation_videos/accident/__WFqm4i3vE_00.mp4"), 40)
