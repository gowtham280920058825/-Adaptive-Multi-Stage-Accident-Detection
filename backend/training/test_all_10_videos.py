import sys
from pathlib import Path
import cv2
import numpy as np
import time

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from backend.detection.yolov11_amf import YOLO11AMFDetector
from backend.detection.temporal_engine import TemporalEngine
from backend.detection.video_processor import VideoProcessor
from backend.config import CONFIDENCE_THRESHOLD, YOLO_AMF_WEIGHTS, RESULTS_DIR

detector = YOLO11AMFDetector(weights_path=YOLO_AMF_WEIGHTS, conf_thresh=CONFIDENCE_THRESHOLD)

def is_ego_border_box(box, img_w, img_h):
    """Checks if detection box is an ego-vehicle hood / dashboard / side-mirror artifact at camera boundary."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    # Touches bottom edge and corner
    touches_bottom = (y2 >= img_h - 10)
    touches_left = (x1 <= 15)
    touches_right = (x2 >= img_w - 15)
    if touches_bottom and (touches_left or touches_right) and (h > img_h * 0.4):
        return True
    return False

def evaluate_amave_candidate(dets, history, frames_cache, img_shape):
    """
    Enhanced AMAVE 5-Stage Verification Logic
    """
    if not dets:
        return {
            "verified": False,
            "fused": 0.0,
            "s_det": 0.0,
            "s_temp": 0.0,
            "s_motion": 0.0,
            "s_inter": 0.0,
            "reason": "No detections"
        }

    img_h, img_w = img_shape[:2]
    
    # Filter out obvious ego-vehicle corner/hood artifacts
    valid_dets = [d for d in dets if not is_ego_border_box(d["box"], img_w, img_h)]
    if not valid_dets:
        return {
            "verified": False,
            "fused": 0.20,
            "s_det": max([d["confidence"] for d in dets], default=0.0),
            "s_temp": 0.0,
            "s_motion": 0.20,
            "s_inter": 0.15,
            "reason": "Suppressed ego-vehicle border artifact"
        }

    # Stage 1: Detection confidence
    s_det = float(max([d["confidence"] for d in valid_dets], default=0.0))

    # Stage 2: Temporal persistence & Kinematics
    consecutive = 0
    max_consecutive = 0
    total_dets = 0
    centroids = []
    
    for h in history:
        if h.get("has_candidate", False):
            # Check if any non-ego detection
            v_dets = [d for d in h["detections"] if not is_ego_border_box(d["box"], img_w, img_h)]
            if v_dets:
                consecutive += 1
                total_dets += 1
                max_consecutive = max(max_consecutive, consecutive)
                b = max(v_dets, key=lambda d: d["confidence"])["box"]
                centroids.append(np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]))
            else:
                consecutive = 0
        else:
            consecutive = 0

    window_len = len(history)
    pers_ratio = total_dets / window_len if window_len > 0 else 0.0
    passed_pers = (max_consecutive >= 2) or (pers_ratio >= 0.40 and window_len >= 5 and total_dets >= 2)
    s_temp = pers_ratio

    # Trajectory & Deceleration Analysis
    has_deceleration = False
    is_stationary_post_impact = False
    if len(centroids) >= 3:
        disps = [np.linalg.norm(centroids[i] - centroids[i-1]) for i in range(1, len(centroids))]
        recent_disp = float(np.mean(disps[-2:]))
        if len(disps) >= 4:
            early_disp = float(np.mean(disps[:2]))
            if early_disp > 6.0 and recent_disp < 4.5:
                has_deceleration = True
        # If vehicle was moving and is now stopped, or is persistently stopped on active roadway
        if recent_disp < 5.0 and pers_ratio >= 0.50:
            is_stationary_post_impact = True

    # Stage 3: Motion Consistency with Local-vs-Global Background Ring Estimation
    s_motion = 0.35
    has_kinetic_disruption = False
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
        for d in valid_dets:
            box = d["box"]
            x1, y1 = max(0, int(box[0] * sx)), max(0, int(box[1] * sy))
            x2, y2 = min(w_proc, int(box[2] * sx)), min(h_proc, int(box[3] * sy))
            if x2 <= x1 or y2 <= y1:
                continue

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
            local_discontinuity = float(np.linalg.norm([np.mean(u[y1:y2, x1:x2]) - bg_u, np.mean(v[y1:y2, x1:x2]) - bg_v]))

            # Directional vector coherence (ratio of net vector magnitude to mean magnitude)
            net_mag = float(np.linalg.norm([np.mean(b_u), np.mean(b_v)]))
            coherence = net_mag / (mean_mag + 1e-6)

            # Classify motion:
            # 1. Chaotic turbulent collision kinetics: low coherence (cancelling/rotating vectors) with high dispersion
            if (coherence < 0.75 and std_mag > 0.65 and mean_mag > 0.4) or \
               (angle_var > 1.1 and std_mag > 0.65 and mean_mag > 0.4):
                b_score = 0.70  # Chaotic turbulent collision kinetics
                has_kinetic_disruption = True
            elif local_discontinuity > 0.8 and (has_deceleration or is_stationary_post_impact):
                b_score = 0.65  # Post-impact stoppage with background motion discontinuity
                has_kinetic_disruption = True
            elif (has_deceleration or is_stationary_post_impact) and s_det >= 0.35:
                b_score = 0.55  # Confirmed post-impact deceleration/stationary hazard
                has_kinetic_disruption = True
            elif coherence >= 0.82 or (mean_mag < 0.35 and local_discontinuity < 0.50):
                b_score = 0.20  # Smooth laminar vehicular flow (unidirectional rigid motion)
            else:
                b_score = 0.35

            box_scores.append(b_score)

        s_motion = float(np.mean(box_scores)) if box_scores else 0.35

    # Stage 4: Vehicle Interaction
    # Deduplicate
    sorted_dets = sorted(valid_dets, key=lambda d: d.get("confidence", 0.0), reverse=True)
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
        # Check aspect ratio for rollover / severe deformation
        box = keep[0]["box"]
        w = abs(box[2] - box[0])
        h = abs(box[3] - box[1])
        aspect_ratio = w / (h + 1e-6)
        is_edge = (box[0] <= 15 or box[1] <= 15 or box[2] >= img_w - 15 or box[3] >= img_h - 15)
        if not is_edge and (aspect_ratio < 0.25 or aspect_ratio > 4.5):
            s_inter = 0.60
        else:
            s_inter = 0.15

    # Stage 5: Confidence Fusion
    w_det, w_temp, w_motion, w_inter = 0.30, 0.20, 0.30, 0.20
    fused_score = (w_det * s_det) + (w_temp * s_temp) + (w_motion * s_motion) + (w_inter * s_inter)

    # Collision evidence criteria:
    # 1. Multi-vehicle physical contact interaction (s_inter >= 0.35, s_det >= 0.20)
    # 2. OR confirmed kinetic disruption / post-impact stoppage (s_motion >= 0.50, s_det >= 0.25, s_temp >= 0.40)
    # NO confidence-alone bypass!
    has_collision_evidence = (
        (s_inter >= 0.35 and s_det >= 0.20) or
        (s_motion >= 0.50 and s_det >= 0.25 and s_temp >= 0.40)
    )

    is_verified = (fused_score >= 0.45) and passed_pers and has_collision_evidence
    
    return {
        "verified": is_verified,
        "fused": round(float(fused_score), 4),
        "s_det": round(s_det, 4),
        "s_temp": round(s_temp, 4),
        "s_motion": round(s_motion, 4),
        "s_inter": round(s_inter, 4),
        "passed_pers": passed_pers,
        "has_collision_evidence": has_collision_evidence
    }

def run_all_benchmark():
    eval_dir = BASE_DIR / "dataset" / "evaluation_videos"
    acc_dir = eval_dir / "accident"
    norm_dir = eval_dir / "non_accident"
    
    acc_videos = sorted(list(acc_dir.glob("*.mp4")))
    norm_videos = sorted(list(norm_dir.glob("*.mp4")))
    
    print(f"\n=======================================================")
    print(f"BENCHMARKING ENHANCED AMAVE ON {len(acc_videos)} ACCIDENT & {len(norm_videos)} NON-ACCIDENT VIDEOS")
    print(f"=======================================================\n")
    
    results = []
    
    def test_video(v_path, gt):
        vp = VideoProcessor(v_path)
        temp_engine = TemporalEngine(window_size=10)
        frames_cache = []
        frame_results = []
        max_yolo_conf = 0.0
        
        for frame_idx, timestamp, frame in vp.extract_frames(stride=2):
            frames_cache.append(frame)
            dets = detector.detect_frame(frame)
            temp_engine.push_frame_detections(frame_idx, timestamp, dets)
            if dets:
                max_yolo_conf = max(max_yolo_conf, max([d["confidence"] for d in dets]))
                
            history = list(temp_engine.history)
            seq = frames_cache[max(0, len(frames_cache)-2):]
            
            res = evaluate_amave_candidate(dets, history, seq, frame.shape)
            frame_results.append(res)
            
        vp.release()
        
        verified_count = sum(1 for r in frame_results if r["verified"])
        peak_fused = max([r["fused"] for r in frame_results], default=0.0)
        is_accident = (verified_count >= 2)
        prediction = "ACCIDENT DETECTED" if is_accident else "NO ACCIDENT DETECTED"
        correct = (prediction == ("ACCIDENT DETECTED" if gt == "ACCIDENT" else "NO ACCIDENT DETECTED"))
        
        return {
            "video": v_path.name,
            "gt": gt,
            "max_yolo": round(max_yolo_conf, 4),
            "peak_fused": round(peak_fused, 4),
            "verified_frames": verified_count,
            "prediction": prediction,
            "correct": correct
        }

    for v in acc_videos:
        res = test_video(v, "ACCIDENT")
        results.append(res)
        print(f"ACCIDENT:     {res['video']:25s} | MaxYOLO={res['max_yolo']:.2f} PeakFused={res['peak_fused']:.2f} Verified={res['verified_frames']:2d} -> {res['prediction']:20s} {'[OK]' if res['correct'] else '[FAIL]'}")

    for v in norm_videos:
        res = test_video(v, "NO ACCIDENT")
        results.append(res)
        print(f"NON-ACCIDENT: {res['video']:25s} | MaxYOLO={res['max_yolo']:.2f} PeakFused={res['peak_fused']:.2f} Verified={res['verified_frames']:2d} -> {res['prediction']:20s} {'[OK]' if res['correct'] else '[FAIL]'}")

    tp = sum(1 for r in results if r["gt"] == "ACCIDENT" and r["correct"])
    fn = sum(1 for r in results if r["gt"] == "ACCIDENT" and not r["correct"])
    tn = sum(1 for r in results if r["gt"] == "NO ACCIDENT" and r["correct"])
    fp = sum(1 for r in results if r["gt"] == "NO ACCIDENT" and not r["correct"])

    acc = (tp + tn) / len(results) * 100
    prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0.0

    print(f"\n=======================================================")
    print(f"RESULTS SUMMARY:")
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Accuracy:  {acc:.2f}%")
    print(f"Precision: {prec:.2f}%")
    print(f"Recall:    {rec:.2f}%")
    print(f"F1 Score:  {f1:.2f}%")
    print(f"FPR:       {fpr:.2f}%")
    print(f"=======================================================\n")

if __name__ == "__main__":
    run_all_benchmark()
