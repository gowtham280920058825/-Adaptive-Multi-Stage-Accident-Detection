import sys
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from backend.detection.yolov11_amf import YOLO11AMFDetector
from backend.detection.temporal_engine import TemporalEngine
from backend.detection.video_processor import VideoProcessor
from backend.config import CONFIDENCE_THRESHOLD, YOLO_AMF_WEIGHTS

detector = YOLO11AMFDetector(weights_path=YOLO_AMF_WEIGHTS, conf_thresh=CONFIDENCE_THRESHOLD)

def analyze_prototype(video_path):
    print(f"\n=======================================================")
    print(f"PROTOTYPE TEST: {video_path.name}")
    print(f"=======================================================")
    
    vp = VideoProcessor(video_path)
    temp_engine = TemporalEngine(window_size=10)
    frames_cache = []
    
    verified_frames = []
    all_fused = []
    max_det = 0.0
    
    for frame_idx, timestamp, frame in vp.extract_frames(stride=2):
        frames_cache.append(frame)
        dets = detector.detect_frame(frame)
        temp_engine.push_frame_detections(frame_idx, timestamp, dets)
        
        if not dets:
            continue
            
        frame_max_conf = max([d["confidence"] for d in dets], default=0.0)
        max_det = max(max_det, frame_max_conf)
        
        # 1. Temporal Persistence & Kinematic History
        history = list(temp_engine.history)
        consecutive = 0
        max_consecutive = 0
        total_dets = 0
        centroids = []
        for h in history:
            if h.get("has_candidate", False):
                consecutive += 1
                total_dets += 1
                max_consecutive = max(max_consecutive, consecutive)
                b = max(h["detections"], key=lambda d: d["confidence"])["box"]
                centroids.append(np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]))
            else:
                consecutive = 0
                
        window_len = len(history)
        pers_ratio = total_dets / window_len if window_len > 0 else 0.0
        passed_pers = (max_consecutive >= 2) or (pers_ratio >= 0.40 and window_len >= 5 and total_dets >= 2)
        s_temp = pers_ratio
        
        # Deceleration / Stoppage analysis from centroids
        has_deceleration = False
        is_stationary_hazard = False
        if len(centroids) >= 3:
            disps = [np.linalg.norm(centroids[i] - centroids[i-1]) for i in range(1, len(centroids))]
            recent_disp = np.mean(disps[-2:])
            if len(disps) >= 4:
                early_disp = np.mean(disps[:2])
                if early_disp > 7.0 and recent_disp < 4.0:
                    has_deceleration = True
            if recent_disp < 3.5 and pers_ratio >= 0.60:
                is_stationary_hazard = True

        # 2. Motion Consistency with Local-vs-Global Background Estimation
        s_motion = 0.35
        motion_disruption = False
        if len(frames_cache) >= 2:
            f1, f2 = frames_cache[-2], frames_cache[-1]
            h_orig, w_orig = f1.shape[:2]
            h_proc, w_proc = 240, 320
            f1_s = cv2.resize(f1, (w_proc, h_proc))
            f2_s = cv2.resize(f2, (w_proc, h_proc))
            g1 = cv2.cvtColor(f1_s, cv2.COLOR_BGR2GRAY) if len(f1_s.shape) == 3 else f1_s
            g2 = cv2.cvtColor(f2_s, cv2.COLOR_BGR2GRAY) if len(f2_s.shape) == 3 else f2_s
            
            flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 2, 11, 2, 5, 1.1, 0)
            u, v = flow[..., 0], flow[..., 1]
            
            sx = w_proc / max(1.0, float(w_orig))
            sy = h_proc / max(1.0, float(h_orig))
            
            box_scores = []
            for d in dets:
                box = d["box"]
                x1, y1 = max(0, int(box[0] * sx)), max(0, int(box[1] * sy))
                x2, y2 = min(w_proc, int(box[2] * sx)), min(h_proc, int(box[3] * sy))
                if x2 <= x1 or y2 <= y1:
                    continue
                
                bw, bh = x2 - x1, y2 - y1
                # Local background context ring (expanded 25% around candidate box)
                pad_x = int(bw * 0.25)
                pad_y = int(bh * 0.25)
                rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                rx2, ry2 = min(w_proc, x2 + pad_x), min(h_proc, y2 + pad_y)
                
                ring_u = u[ry1:ry2, rx1:rx2]
                ring_v = v[ry1:ry2, rx1:rx2]
                
                # Mask out box itself from ring
                mask = np.ones((ry2 - ry1, rx2 - rx1), dtype=bool)
                bx1, by1 = x1 - rx1, y1 - ry1
                bx2, by2 = bx1 + bw, by1 + bh
                mask[by1:by2, bx1:bx2] = False
                
                if np.any(mask):
                    bg_u = float(np.median(ring_u[mask]))
                    bg_v = float(np.median(ring_v[mask]))
                else:
                    bg_u = float(np.median(u))
                    bg_v = float(np.median(v))
                    
                b_u = u[y1:y2, x1:x2] - bg_u
                b_v = v[y1:y2, x1:x2] - bg_v
                b_mag = np.sqrt(b_u**2 + b_v**2)
                
                mean_mag = float(np.mean(b_mag))
                std_mag = float(np.std(b_mag))
                angles = np.arctan2(b_v, b_u)
                angle_var = float(np.var(angles))
                
                # Motion discontinuity relative to local roadway
                local_discontinuity = float(np.linalg.norm([np.mean(u[y1:y2, x1:x2]) - bg_u, np.mean(v[y1:y2, x1:x2]) - bg_v]))
                
                # Kinetic crash disruption
                if angle_var > 1.2 and std_mag > 0.70 and mean_mag > 0.5:
                    b_score = 0.70
                    motion_disruption = True
                elif local_discontinuity > 1.0 and (has_deceleration or is_stationary_hazard):
                    b_score = 0.65  # Post-impact stoppage with background discontinuity
                    motion_disruption = True
                elif is_stationary_hazard and frame_max_conf >= 0.35:
                    b_score = 0.60  # Confirmed post-impact stationary hazard on roadway
                    motion_disruption = True
                elif angle_var < 0.6 or (mean_mag < 0.3 and local_discontinuity < 0.5):
                    b_score = 0.20  # Laminar vehicular flow or camera-following vehicle
                else:
                    b_score = 0.35
                box_scores.append(b_score)
                
            s_motion = float(np.mean(box_scores)) if box_scores else 0.35

        # 3. Vehicle Interaction (Stage 4)
        # Deduplicate
        sorted_dets = sorted(dets, key=lambda d: d.get("confidence", 0.0), reverse=True)
        keep = []
        for det in sorted_dets:
            overlap = False
            for kept in keep:
                b1, b2 = det["box"], kept["box"]
                ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                a1 = abs(b1[2] - b1[0]) * abs(b1[3] - b1[1])
                a2 = abs(b2[2] - b2[0]) * abs(b2[3] - b2[1])
                iou = inter / (a1 + a2 - inter + 1e-7)
                if iou > 0.40:
                    overlap = True
                    break
            if not overlap:
                keep.append(det)
                
        if len(keep) >= 2:
            max_iou = 0.0
            for i in range(len(keep)):
                for j in range(i + 1, len(keep)):
                    b1, b2 = keep[i]["box"], keep[j]["box"]
                    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    a1 = abs(b1[2] - b1[0]) * abs(b1[3] - b1[1])
                    a2 = abs(b2[2] - b2[0]) * abs(b2[3] - b2[1])
                    iou = inter / (a1 + a2 - inter + 1e-7)
                    max_iou = max(max_iou, iou)
            if max_iou > 0.08:
                s_inter = round(min(1.0, 0.60 + max_iou * 0.40), 4)
            else:
                s_inter = 0.20
        else:
            s_inter = 0.15
            
        # 4. Confidence Fusion
        # Weights: det=0.35, temp=0.25, motion=0.20, inter=0.20
        fused = (0.35 * frame_max_conf) + (0.25 * s_temp) + (0.20 * s_motion) + (0.20 * s_inter)
        all_fused.append(fused)
        
        # Collision evidence requirement:
        # Multi-vehicle interaction (s_inter >= 0.35)
        # OR confirmed kinetic impact / motion disruption (s_motion >= 0.50) with persistence
        # NO automatic bypass from confidence alone!
        has_collision_evidence = (
            (s_inter >= 0.35 and frame_max_conf >= 0.20) or
            (s_motion >= 0.50 and frame_max_conf >= 0.25 and s_temp >= 0.40)
        )
        
        is_verified = (fused >= 0.45) and passed_pers and has_collision_evidence
        if is_verified:
            verified_frames.append(frame_idx)
            print(f"  Frame {frame_idx:3d}: VERIFIED! fused={fused:.4f} (det={frame_max_conf:.2f}, temp={s_temp:.2f}, mot={s_motion:.2f}, int={s_inter:.2f})")
            
    vp.release()
    
    is_acc = len(verified_frames) >= 2
    peak_score = max(all_fused) if all_fused else 0.0
    print(f"Summary for {video_path.name}:")
    print(f"  Max YOLO Conf: {max_det:.4f}")
    print(f"  Peak Fused:    {peak_score:.4f}")
    print(f"  Verified Frms: {len(verified_frames)}")
    print(f"  Result:        {'ACCIDENT DETECTED' if is_acc else 'NO ACCIDENT DETECTED'}")
    return is_acc, peak_score, len(verified_frames)

if __name__ == "__main__":
    analyze_prototype(Path("dataset/evaluation_videos/non_accident/normal_traffic_clear.mp4"))
    analyze_prototype(Path("dataset/evaluation_videos/non_accident/normal_traffic_urban.mp4"))
    analyze_prototype(Path("dataset/evaluation_videos/non_accident/normal_traffic_01.mp4"))
    analyze_prototype(Path("dataset/evaluation_videos/accident/nAXTthLfgtI_00.mp4"))
    analyze_prototype(Path("dataset/evaluation_videos/accident/__WFqm4i3vE_00.mp4"))
