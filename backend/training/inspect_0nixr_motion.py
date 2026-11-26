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

vp = VideoProcessor(Path("dataset/evaluation_videos/accident/0NixrF6Mt00_0_00.mp4"))
frames = []
for frame_idx, timestamp, frame in vp.extract_frames(stride=2):
    frames.append(frame)
    dets = detector.detect_frame(frame)
    if dets and 24 <= frame_idx <= 44:
        f1, f2 = frames[-2], frames[-1]
        h_orig, w_orig = f1.shape[:2]
        h_proc, w_proc = 240, 320
        f1_s = cv2.resize(f1, (w_proc, h_proc))
        f2_s = cv2.resize(f2, (w_proc, h_proc))
        g1 = cv2.cvtColor(f1_s, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(f2_s, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 2, 11, 2, 5, 1.1, 0)
        u, v = flow[..., 0], flow[..., 1]
        sx, sy = w_proc / max(1.0, float(w_orig)), h_proc / max(1.0, float(h_orig))
        for d in dets:
            box = d["box"]
            x1, y1 = max(0, int(box[0] * sx)), max(0, int(box[1] * sy))
            x2, y2 = min(w_proc, int(box[2] * sx)), min(h_proc, int(box[3] * sy))
            bw, bh = x2 - x1, y2 - y1
            pad_x = int(bw * 0.25)
            pad_y = int(bh * 0.25)
            rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            rx2, ry2 = min(w_proc, x2 + pad_x), min(h_proc, y2 + pad_y)
            ring_u = u[ry1:ry2, rx1:rx2]
            ring_v = v[ry1:ry2, rx1:rx2]
            mask = np.ones((ry2 - ry1, rx2 - rx1), dtype=bool)
            bx1, by1 = x1 - rx1, y1 - ry1
            bx2, by2 = bx1 + bw, by1 + bh
            mask[by1:by2, bx1:bx2] = False
            bg_u = float(np.median(ring_u[mask])) if np.any(mask) else float(np.median(u))
            bg_v = float(np.median(ring_v[mask])) if np.any(mask) else float(np.median(v))
            b_u = u[y1:y2, x1:x2] - bg_u
            b_v = v[y1:y2, x1:x2] - bg_v
            b_mag = np.sqrt(b_u**2 + b_v**2)
            mean_mag = float(np.mean(b_mag))
            std_mag = float(np.std(b_mag))
            angles = np.arctan2(b_v, b_u)
            angle_var = float(np.var(angles))
            local_disc = float(np.linalg.norm([np.mean(u[y1:y2, x1:x2]) - bg_u, np.mean(v[y1:y2, x1:x2]) - bg_v]))
            print(f"F{frame_idx:2d} conf={d['confidence']:.2f}: mean_mag={mean_mag:.2f} std_mag={std_mag:.2f} angle_var={angle_var:.2f} disc={local_disc:.2f}")
vp.release()
