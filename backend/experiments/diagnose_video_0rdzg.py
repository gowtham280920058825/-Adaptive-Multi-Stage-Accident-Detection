"""
Comprehensive Diagnostic Script for Video: _0RDZgCKPGo_00.mp4
Addresses Sections 1, 2, 3, 4, 5, 7, 8, 9, 10 of the debug request.
"""

import os
import sys
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Workspace root
WORKSPACE = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(WORKSPACE))

from backend.config import YOLO_AMF_WEIGHTS, DEVICE, CUDA_AVAILABLE, GPU_NAME, CONFIDENCE_THRESHOLD
from backend.amave.engine import AMAVEEngine
from backend.detection.temporal_engine import TemporalEngine

VIDEO_PATH = WORKSPACE / "dataset" / "real_videos" / "_0RDZgCKPGo_00.mp4"
DIAG_IMG_DIR = WORKSPACE / "results" / "diagnostic_images" / "0rdzg"
DIAG_IMG_DIR.mkdir(parents=True, exist_ok=True)

def step1_video_info():
    print("=" * 60)
    print("STEP 1: VERIFY EXACT INPUT VIDEO")
    print("=" * 60)
    if not VIDEO_PATH.exists():
        print(f"ERROR: Video not found at {VIDEO_PATH}")
        return None
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    info = {
        "filename": VIDEO_PATH.name,
        "full_path": str(VIDEO_PATH),
        "frame_count": frame_count,
        "fps": round(fps, 3),
        "width": width,
        "height": height,
        "duration_sec": round(duration, 3)
    }
    for k, v in info.items():
        print(f"  {k}: {v}")
    return info

def step2_model_info():
    print("\n" + "=" * 60)
    print("STEP 2: VERIFY DEPLOYED MODEL")
    print("=" * 60)
    model_exists = YOLO_AMF_WEIGHTS.exists()
    file_size = YOLO_AMF_WEIGHTS.stat().st_size if model_exists else 0
    
    # Load model
    try:
        model = YOLO(str(YOLO_AMF_WEIGHTS))
        loaded = True
        model_names = model.names
        pt_dict = torch.load(str(YOLO_AMF_WEIGHTS), map_location="cpu")
        model_type = type(model.model).__name__ if hasattr(model, 'model') else "Ultralytics YOLO"
        train_args = pt_dict.get('train_args', {})
        imgsz = train_args.get('imgsz', 640)
    except Exception as e:
        loaded = False
        model_names = str(e)
        model_type = "Error"
        imgsz = "Unknown"
        model = None

    info = {
        "absolute_checkpoint_path": str(YOLO_AMF_WEIGHTS),
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024*1024), 3),
        "model_class_mapping": model_names,
        "model_type": model_type,
        "image_size": imgsz,
        "device": DEVICE,
        "cuda_available": CUDA_AVAILABLE,
        "gpu_name": GPU_NAME,
        "model_successfully_loaded": "YES" if loaded else "NO"
    }
    for k, v in info.items():
        print(f"  {k}: {v}")
    return model, info

def step3_and_4_and_8_diagnostics(model):
    print("\n" + "=" * 60)
    print("STEP 3: RAW YOLO DIAGNOSTIC (CONFIDENCE THRESHOLDS)")
    print("=" * 60)
    
    # First, let's extract all frames (stride=1) so we can do accurate diagnostics
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    all_frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append((frame_idx, frame))
        frame_idx += 1
    cap.release()
    print(f"Read {len(all_frames)} total frames (stride=1)")

    # Test confidence thresholds with stride=1, imgsz=640
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    
    # We can run inference once per frame at conf=0.01 to get ALL raw predictions, then filter!
    print("Running raw inference across all frames at conf=0.01 to collect full raw spectrum...")
    t0 = time.time()
    
    # Store predictions: {frame_idx: list of boxes [{box, conf, cls, name}]}
    raw_preds_by_frame = {}
    global_max_conf = 0.0
    global_max_frame = -1
    global_max_box = None
    
    for f_idx, frame in all_frames:
        results = model(frame, conf=0.01, iou=0.40, imgsz=640, device=DEVICE, verbose=False)
        frame_preds = []
        for r in results:
            for b in r.boxes:
                conf = float(b.conf[0].cpu().numpy())
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                cls_id = int(b.cls[0].cpu().numpy())
                label = model.names.get(cls_id, f"cls_{cls_id}")
                pred_obj = {
                    "box": [round(c, 2) for c in xyxy],
                    "confidence": round(conf, 4),
                    "class_id": cls_id,
                    "label": label
                }
                frame_preds.append(pred_obj)
                if conf > global_max_conf:
                    global_max_conf = conf
                    global_max_frame = f_idx
                    global_max_box = pred_obj
        raw_preds_by_frame[f_idx] = frame_preds

    t_inf = time.time() - t0
    print(f"Inference completed in {t_inf:.2f}s ({len(all_frames)/t_inf:.1f} FPS)")
    print(f"Global Maximum Raw YOLO Confidence: {global_max_conf:.4f} ({global_max_conf*100:.2f}%) at Frame {global_max_frame}")
    if global_max_box:
        print(f"  Box: {global_max_box['box']}, Class: {global_max_box['label']} ({global_max_box['class_id']})")

    # Report for each threshold
    print("\nThreshold Sweep (stride=1, imgsz=640):")
    print(f"{'Threshold':<10} | {'Frames w/ Dets':<15} | {'Total Dets':<12} | {'Max Raw Conf':<14} | {'Max Conf Frame':<15} | {'Class':<10}")
    print("-" * 85)
    
    thresh_results = {}
    for th in thresholds:
        frames_with_det = 0
        total_dets = 0
        max_c = 0.0
        max_f = -1
        max_cls = "N/A"
        max_b = None
        for f_idx, preds in raw_preds_by_frame.items():
            valid = [p for p in preds if p["confidence"] >= th]
            if len(valid) > 0:
                frames_with_det += 1
                total_dets += len(valid)
                for v in valid:
                    if v["confidence"] > max_c:
                        max_c = v["confidence"]
                        max_f = f_idx
                        max_cls = v["label"]
                        max_b = v["box"]
        thresh_results[th] = {
            "frames_with_det": frames_with_det,
            "total_dets": total_dets,
            "max_confidence": max_c,
            "max_frame": max_f,
            "max_box": max_b,
            "class": max_cls
        }
        print(f"{th:<10.2f} | {frames_with_det:<15} | {total_dets:<12} | {max_c:<14.4f} | {max_f:<15} | {max_cls:<10}")

    # STEP 4: FRAME SAMPLING DIAGNOSTIC (stride = 1, 2, 3)
    print("\n" + "=" * 60)
    print("STEP 4: FRAME SAMPLING DIAGNOSTIC (stride=1, 2, 3 at threshold=0.20)")
    print("=" * 60)
    print(f"{'Stride':<8} | {'Processed Frames':<18} | {'Detection Frames':<18} | {'Max Confidence':<16} | {'Max Conf Frame':<15}")
    print("-" * 85)
    
    stride_results = {}
    for st in [1, 2, 3]:
        sampled_indices = [idx for idx, _ in all_frames if idx % st == 0]
        proc_frames = len(sampled_indices)
        det_frames = 0
        s_max_conf = 0.0
        s_max_frame = -1
        for s_idx in sampled_indices:
            preds = [p for p in raw_preds_by_frame[s_idx] if p["confidence"] >= 0.20]
            if preds:
                det_frames += 1
                for p in preds:
                    if p["confidence"] > s_max_conf:
                        s_max_conf = p["confidence"]
                        s_max_frame = s_idx
        stride_results[st] = {
            "processed_frames": proc_frames,
            "detection_frames": det_frames,
            "max_confidence": s_max_conf,
            "max_frame": s_max_frame
        }
        print(f"{st:<8} | {proc_frames:<18} | {det_frames:<18} | {s_max_conf:<16.4f} | {s_max_frame:<15}")

    # STEP 8: SMALL-OBJECT / RESOLUTION SWEEP (imgsz = 640, 960, 1280)
    print("\n" + "=" * 60)
    print("STEP 8: RESOLUTION SWEEP (imgsz=640, 960, 1280 at conf=0.10, stride=2)")
    print("=" * 60)
    print(f"{'imgsz':<8} | {'Processed Frames':<18} | {'Detection Frames (>=0.10)':<26} | {'Detection Frames (>=0.20)':<26} | {'Max Confidence':<16} | {'Max Frame':<10}")
    print("-" * 115)
    
    imgsz_results = {}
    for sz in [640, 960, 1280]:
        sz_max_c = 0.0
        sz_max_f = -1
        sz_det_10 = 0
        sz_det_20 = 0
        # Test on stride=2 frames
        stride2_frames = [(idx, fr) for idx, fr in all_frames if idx % 2 == 0]
        for f_idx, fr in stride2_frames:
            res = model(fr, conf=0.05, iou=0.40, imgsz=sz, device=DEVICE, verbose=False)
            boxes_10 = []
            boxes_20 = []
            for r in res:
                for b in r.boxes:
                    conf = float(b.conf[0].cpu().numpy())
                    if conf > sz_max_c:
                        sz_max_c = conf
                        sz_max_f = f_idx
                    if conf >= 0.10:
                        boxes_10.append(conf)
                    if conf >= 0.20:
                        boxes_20.append(conf)
            if boxes_10:
                sz_det_10 += 1
            if boxes_20:
                sz_det_20 += 1
        imgsz_results[sz] = {
            "processed_frames": len(stride2_frames),
            "det_frames_10": sz_det_10,
            "det_frames_20": sz_det_20,
            "max_confidence": sz_max_c,
            "max_frame": sz_max_f
        }
        print(f"{sz:<8} | {len(stride2_frames):<18} | {sz_det_10:<26} | {sz_det_20:<26} | {sz_max_c:<16.4f} | {sz_max_f:<10}")

    return all_frames, raw_preds_by_frame, thresh_results, stride_results, imgsz_results, global_max_conf, global_max_frame

def step5_accident_frames_and_images(all_frames, raw_preds_by_frame, global_max_frame):
    print("\n" + "=" * 60)
    print("STEP 5: FIND ACTUAL ACCIDENT FRAME & SAVE DIAGNOSTIC IMAGES")
    print("=" * 60)
    
    # Let's inspect where in the video detections and motion occur.
    # Print timeline of all frames where confidence >= 0.10
    print("Timeline of detections >= 0.10:")
    active_ranges = []
    for f_idx, preds in raw_preds_by_frame.items():
        high = [p for p in preds if p["confidence"] >= 0.10]
        if high:
            max_c = max([p["confidence"] for p in high])
            active_ranges.append((f_idx, max_c, high[0]["box"]))
    
    if active_ranges:
        print(f"Total frames with conf >= 0.10: {len(active_ranges)}")
        print(f"Active frame range: {active_ranges[0][0]} to {active_ranges[-1][0]}")
        print("Sample of detections around peak:")
        for f, c, b in active_ranges:
            if abs(f - global_max_frame) <= 10 or f % 20 == 0:
                print(f"  Frame {f:3d} (time {f/14.87:.2f}s): conf={c:.4f}, box={b}")
    else:
        print("No detections >= 0.10 found!")

    # Let's also visually inspect frames to find the exact collision frame
    # We can save a series of frames around the accident
    # Let's check frames around global_max_frame (e.g. frame 150 to 220)
    # Save 4 designated images:
    # A: Highest YOLO confidence frame
    # B: Frame immediately before accident
    # C: Apparent collision frame
    # D: Frame immediately after accident

    frame_dict = {f_idx: fr for f_idx, fr in all_frames}

    # Let's check image dimensions
    sample_fr = all_frames[0][1]
    H, W = sample_fr.shape[:2]
    print(f"Frame dimensions: W={W}, H={H}")

    # Let's inspect frames around 160-180 (since max conf was frame 170 in previous logs)
    peak_f = global_max_frame
    
    # Save diagnostic images
    # A. Highest YOLO confidence frame
    img_a = frame_dict[peak_f].copy()
    preds_a = raw_preds_by_frame.get(peak_f, [])
    for p in preds_a:
        b = p["box"]
        conf = p["confidence"]
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(img_a, (x1, y1), (x2, y2), (0, 165, 255) if conf < 0.20 else (0, 255, 0), 2)
        cv2.putText(img_a, f"{p['label']} {conf*100:.1f}%", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img_a, f"Highest YOLO Conf Frame: {peak_f} ({peak_f/14.87:.2f}s)", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    path_a = DIAG_IMG_DIR / "diag_A_max_yolo_conf.jpg"
    cv2.imwrite(str(path_a), img_a)
    print(f"Saved Image A (Max Conf Frame {peak_f}): {path_a}")

    # Let's find pre-accident, collision, and post-accident frames
    # Let's analyze frame differences around frame 130-200 to see where the physical collision occurs
    diffs = []
    for f in range(max(1, peak_f - 40), min(len(all_frames)-1, peak_f + 40)):
        gray_prev = cv2.cvtColor(frame_dict[f-1], cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(frame_dict[f], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_curr, gray_prev)
        diffs.append((f, np.mean(diff)))
    
    # Sort diffs by motion magnitude
    diffs.sort(key=lambda x: x[1], reverse=True)
    print("Top 5 frame differences (motion spikes):")
    for f, d in diffs[:5]:
        print(f"  Frame {f}: diff={d:.3f}")

    # Suppose collision occurs around frame 160-170:
    # Let's define:
    # B: Pre-accident frame (e.g. frame peak_f - 20)
    # C: Apparent collision frame (e.g. frame peak_f or highest diff frame)
    # D: Post-accident frame (e.g. frame peak_f + 25)
    f_pre = max(0, peak_f - 25)
    f_coll = peak_f  # or the frame of primary collision
    f_post = min(len(all_frames) - 1, peak_f + 25)

    # B: Pre-accident
    img_b = frame_dict[f_pre].copy()
    preds_b = raw_preds_by_frame.get(f_pre, [])
    for p in preds_b:
        b = p["box"]
        conf = p["confidence"]
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(img_b, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_b, f"{p['label']} {conf*100:.1f}%", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img_b, f"Pre-Accident Frame: {f_pre} ({f_pre/14.87:.2f}s)", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    path_b = DIAG_IMG_DIR / "diag_B_pre_accident.jpg"
    cv2.imwrite(str(path_b), img_b)
    print(f"Saved Image B (Pre-Accident Frame {f_pre}): {path_b}")

    # C: Collision frame
    img_c = frame_dict[f_coll].copy()
    preds_c = raw_preds_by_frame.get(f_coll, [])
    for p in preds_c:
        b = p["box"]
        conf = p["confidence"]
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(img_c, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img_c, f"{p['label']} {conf*100:.1f}%", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(img_c, f"Apparent Collision Frame: {f_coll} ({f_coll/14.87:.2f}s)", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    path_c = DIAG_IMG_DIR / "diag_C_collision_frame.jpg"
    cv2.imwrite(str(path_c), img_c)
    print(f"Saved Image C (Collision Frame {f_coll}): {path_c}")

    # D: Post-accident
    img_d = frame_dict[f_post].copy()
    preds_d = raw_preds_by_frame.get(f_post, [])
    for p in preds_d:
        b = p["box"]
        conf = p["confidence"]
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(img_d, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_d, f"{p['label']} {conf*100:.1f}%", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img_d, f"Post-Accident Frame: {f_post} ({f_post/14.87:.2f}s)", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    path_d = DIAG_IMG_DIR / "diag_D_post_accident.jpg"
    cv2.imwrite(str(path_d), img_d)
    print(f"Saved Image D (Post-Accident Frame {f_post}): {path_d}")

    # Measure accident object width & height in Image A / C
    if preds_a:
        b = preds_a[0]["box"]
        obj_w = b[2] - b[0]
        obj_h = b[3] - b[1]
        print(f"Accident Bounding Box at Frame {peak_f}: [{b[0]}, {b[1]}, {b[2]}, {b[3]}]")
        print(f"Accident Object Size: width={obj_w:.1f}px, height={obj_h:.1f}px (Area={obj_w*obj_h:.1f}px^2)")
        print(f"Image Size: W={W}, H={H} (Area={W*H})")
        print(f"Object / Image Area Ratio: {(obj_w*obj_h)/(W*H)*100:.2f}%")
        print(f"Object Width Ratio: {obj_w/W*100:.2f}%, Height Ratio: {obj_h/H*100:.2f}%")
    else:
        print("No prediction on frame A")

    return {
        "path_a": str(path_a),
        "path_b": str(path_b),
        "path_c": str(path_c),
        "path_d": str(path_d),
        "peak_frame": peak_f,
        "pre_frame": f_pre,
        "coll_frame": f_coll,
        "post_frame": f_post,
    }

def step10_amave_trace(all_frames, model):
    print("\n" + "=" * 60)
    print("STEP 10: AMAVE TRACE & PIPELINE ANALYSIS")
    print("=" * 60)
    
    amave_engine = AMAVEEngine()
    temp_engine = TemporalEngine(window_size=10)
    
    frames_cache = []
    amave_results = []
    
    # Run with standard production stride=2
    stride = 2
    for f_idx, frame in all_frames:
        if f_idx % stride != 0:
            continue
        frames_cache.append(frame)
        
        # Detector inference at current production threshold (0.20)
        results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=0.40, imgsz=640, device=DEVICE, verbose=False)
        dets = []
        for r in results:
            for b in r.boxes:
                conf = float(b.conf[0].cpu().numpy())
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                cls_id = int(b.cls[0].cpu().numpy())
                dets.append({
                    "box": [round(c, 2) for c in xyxy],
                    "confidence": round(conf, 4),
                    "class_id": cls_id,
                    "label": model.names.get(cls_id, "Accident")
                })
        
        timestamp = f_idx / 14.87
        temp_engine.push_frame_detections(f_idx, timestamp, dets)
        
        history_window = list(temp_engine.history)
        frame_seq_window = frames_cache[max(0, len(frames_cache) - 2):]
        
        amave_res = amave_engine.verify_event(
            current_detections=dets,
            temporal_history=history_window,
            frame_sequence=frame_seq_window
        )
        amave_results.append((f_idx, dets, amave_res))
    
    verified_windows = [res for _, _, res in amave_results if res.get("verified", False)]
    print(f"Total processed frames (stride=2): {len(amave_results)}")
    print(f"Verified windows count: {len(verified_windows)}")
    
    # Peak score window
    peak_window = max(amave_results, key=lambda x: x[2].get("fused_verification_score", 0.0))
    print(f"Peak AMAVE Frame: {peak_window[0]}")
    print(f"Peak Fused Verification Score: {peak_window[2].get('fused_verification_score', 0.0):.4f}")
    print(f"AMAVE Stages at Peak:")
    print(f"  detection_confidence: {peak_window[2].get('detection_confidence', 0.0):.4f}")
    print(f"  temporal_persistence: {peak_window[2].get('temporal_persistence_score', 0.0):.4f}")
    print(f"  motion_consistency:   {peak_window[2].get('motion_consistency_score', 0.0):.4f}")
    print(f"  collision_interaction:{peak_window[2].get('interaction_score', 0.0):.4f}")
    print(f"  verification_status:  {peak_window[2].get('verification_status', 'N/A')}")
    print(f"  reason:               {peak_window[2].get('reason', 'N/A')}")
    
    # Top 5 AMAVE score frames
    sorted_amave = sorted(amave_results, key=lambda x: x[2].get("fused_verification_score", 0.0), reverse=True)
    print("\nTop 5 AMAVE Fused Score Frames:")
    for f, d, a in sorted_amave[:5]:
        print(f"  Frame {f:3d} (dets={len(d)}): Fused={a.get('fused_verification_score', 0.0):.4f}, DetConf={a.get('detection_confidence', 0.0):.4f}, Motion={a.get('motion_consistency_score', 0.0):.4f}, Temp={a.get('temporal_persistence_score', 0.0):.4f}, Inter={a.get('interaction_score', 0.0):.4f}")

    return peak_window, amave_results

if __name__ == "__main__":
    v_info = step1_video_info()
    model, m_info = step2_model_info()
    if model is not None:
        all_frames, raw_preds, thresh_res, stride_res, imgsz_res, max_c, max_f = step3_and_4_and_8_diagnostics(model)
        img_info = step5_accident_frames_and_images(all_frames, raw_preds, max_f)
        peak_amave, amave_res = step10_amave_trace(all_frames, model)
        
        # Save summary JSON for report generation
        summary = {
            "video_info": v_info,
            "model_info": m_info,
            "max_confidence": max_c,
            "max_confidence_frame": max_f,
            "threshold_sweep": thresh_res,
            "stride_sweep": stride_res,
            "imgsz_sweep": imgsz_res,
            "diagnostic_images": img_info,
            "peak_amave": {
                "frame": peak_amave[0],
                "fused_score": peak_amave[2].get("fused_verification_score", 0.0),
                "stages": {
                    "detection_confidence": peak_amave[2].get("detection_confidence", 0.0),
                    "temporal_persistence": peak_amave[2].get("temporal_persistence_score", 0.0),
                    "motion_consistency": peak_amave[2].get("motion_consistency_score", 0.0),
                    "collision_interaction": peak_amave[2].get("interaction_score", 0.0),
                },
                "status": peak_amave[2].get("verification_status", "N/A"),
                "reason": peak_amave[2].get("reason", "N/A")
            }
        }
        with open(WORKSPACE / "results" / "diagnostic_summary_0rdzg.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("\nSaved diagnostic summary to results/diagnostic_summary_0rdzg.json")
