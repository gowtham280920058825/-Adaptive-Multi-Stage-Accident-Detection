"""
Comprehensive Audit & Diagnostics for YOLO11-AMF
Covers User Mandate Sections 3, 4, 5, 6, 7:
3. Check Class Mapping
4. Check Current Model on Ground-Truth Frames
5. Diagnostic Confidence Threshold Sweep (0.10, 0.20, 0.30, 0.40, 0.50)
6. Check Video Frame Sampling
7. Run Actual Accident Video (__WFqm4i3vE_00.mp4) through YOLO11-AMF Only
"""

import sys
import json
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

def run_audit():
    print("=" * 60)
    print("SECTION 3: CHECK CLASS MAPPING")
    print("=" * 60)

    model_path = BASE_DIR / "backend" / "models" / "yolov11_amf.pt"
    model = YOLO(str(model_path))

    yaml_path = BASE_DIR / "dataset" / "prepared" / "dataset.yaml"
    import yaml
    with open(yaml_path) as f:
        yaml_data = yaml.safe_load(f)

    dataset_class_mapping = yaml_data.get("names", {})
    model_class_mapping = model.names

    print(f"Dataset YAML Class Mapping : {dataset_class_mapping}")
    print(f"Model Checkpoint Class Mapping: {model_class_mapping}")
    print(f"Class 0 Name in YAML       : {dataset_class_mapping.get(0, 'MISSING')}")
    print(f"Class 0 Name in Model      : {model_class_mapping.get(0, 'MISSING')}")
    mapping_aligned = (dataset_class_mapping.get(0) == model_class_mapping.get(0))
    print(f"Class Mapping Aligned      : {'YES' if mapping_aligned else 'NO'}")

    print("\n" + "=" * 60)
    print("SECTION 4: CHECK CURRENT MODEL ON KNOWN LABELED ACCIDENT FRAMES")
    print("=" * 60)

    val_img_dir = BASE_DIR / "dataset" / "prepared" / "images" / "val"
    val_lbl_dir = BASE_DIR / "dataset" / "prepared" / "labels" / "val"

    sample_accident_images = []
    for lbl in sorted(val_lbl_dir.glob("*.txt")):
        with open(lbl) as f:
            content = f.read().strip()
            if content:
                img_name = lbl.stem + ".jpg"
                img_path = val_img_dir / img_name
                if img_path.exists():
                    sample_accident_images.append((img_path, content))
                if len(sample_accident_images) >= 6:
                    break

    for idx, (img_p, gt_str) in enumerate(sample_accident_images):
        img = cv2.imread(str(img_p))
        h, w = img.shape[:2]

        gt_boxes = []
        for line in gt_str.splitlines():
            parts = line.split()
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            gt_boxes.append((cls_id, [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]))

        res = model(img, conf=0.10, iou=0.45, verbose=False)[0]
        preds = []
        for b in res.boxes:
            xyxy = [round(float(c), 1) for c in b.xyxy[0].cpu().numpy()]
            conf = round(float(b.conf[0].cpu().numpy()), 4)
            c_id = int(b.cls[0].cpu().numpy())
            preds.append({"class_id": c_id, "box": xyxy, "conf": conf})

        # Calculate matching
        is_correct = "NO"
        for p in preds:
            px1, py1, px2, py2 = p["box"]
            for g_cls, (gx1, gy1, gx2, gy2) in gt_boxes:
                ix1 = max(px1, gx1)
                iy1 = max(py1, gy1)
                ix2 = min(px2, gx2)
                iy2 = min(py2, gy2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = (px2 - px1) * (py2 - py1) + (gx2 - gx1) * (gy2 - gy1) - inter + 1e-6
                iou = inter / union
                if iou >= 0.30 and p["conf"] >= 0.20:
                    is_correct = "YES"
                    break

        print(f"\nSample Frame {idx+1}: {img_p.name}")
        print(f"Ground truth: {gt_boxes}")
        if preds:
            for p in preds:
                print(f"Prediction: class={p['class_id']} ({model.names[p['class_id']]})")
                print(f"Confidence: {p['conf']}")
                print(f"Bounding box: {p['box']}")
        else:
            print("Prediction: None (No proposals >= 0.10)")
            print("Confidence: 0.0")
            print("Bounding box: None")
        print(f"Correct: {is_correct}")

    print("\n" + "=" * 60)
    print("SECTION 5: DIAGNOSTIC CONFIDENCE THRESHOLD SWEEP")
    print("=" * 60)

    # Evaluate across 100 validation frames at different thresholds
    test_thresholds = [0.10, 0.20, 0.30, 0.40, 0.50]
    val_files = list(val_lbl_dir.glob("*.txt"))[:100]

    sweep_results = {}
    for thresh in test_thresholds:
        tp, fp, fn = 0, 0, 0
        for lbl in val_files:
            img_p = val_img_dir / (lbl.stem + ".jpg")
            if not img_p.exists():
                continue
            img = cv2.imread(str(img_p))
            h, w = img.shape[:2]

            gt_boxes = []
            with open(lbl) as f:
                for line in f.read().strip().splitlines():
                    if not line:
                        continue
                    parts = line.split()
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w
                    y2 = (cy + bh / 2) * h
                    gt_boxes.append([x1, y1, x2, y2])

            res = model(img, conf=thresh, iou=0.45, verbose=False)[0]
            preds = [b.xyxy[0].cpu().numpy().tolist() for b in res.boxes]

            matched_gt = set()
            for p_box in preds:
                px1, py1, px2, py2 = p_box
                matched = False
                for g_idx, (gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
                    if g_idx in matched_gt:
                        continue
                    ix1, iy1 = max(px1, gx1), max(py1, gy1)
                    ix2, iy2 = min(px2, gx2), min(py2, gy2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    union = (px2 - px1)*(py2 - py1) + (gx2 - gx1)*(gy2 - gy1) - inter + 1e-6
                    if inter / union >= 0.30:
                        matched = True
                        matched_gt.add(g_idx)
                        break
                if matched:
                    tp += 1
                else:
                    fp += 1
            fn += len(gt_boxes) - len(matched_gt)

        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        f1 = 2 * prec * rec / (prec + rec + 1e-6)
        sweep_results[thresh] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
        print(f"Threshold {thresh:.2f} -> TP: {tp:3d}, FP: {fp:3d}, FN: {fn:3d} | Precision: {prec*100:5.2f}% | Recall: {rec*100:5.2f}% | F1: {f1*100:5.2f}%")

    print("\n" + "=" * 60)
    print("SECTION 6: VIDEO FRAME SAMPLING AUDIT ON __WFqm4i3vE_00.mp4")
    print("=" * 60)

    target_video_path = BASE_DIR / "results" / "temp_uploads" / "__WFqm4i3vE_00.mp4"
    if not target_video_path.exists():
        target_video_path = BASE_DIR / "dataset" / "real_videos" / "__WFqm4i3vE_00.mp4"

    cap = cv2.VideoCapture(str(target_video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    csv_path = BASE_DIR / "dataset" / "metadata-real.csv"
    df = pd.read_csv(csv_path)
    video_meta = df[df["path"].str.contains("__WFqm4i3vE", na=False)]
    accident_frame_gt = int(video_meta["accident_frame"].values[0]) if len(video_meta) > 0 else 109
    accident_time_gt = float(video_meta["accident_time"].values[0]) if len(video_meta) > 0 else 7.3

    print(f"Total video frames: {total_frames}")
    print(f"Video FPS: {fps:.2f}")
    print(f"Ground truth accident frame in metadata: {accident_frame_gt} (~{accident_time_gt:.2f}s)")

    print("\n" + "=" * 60)
    print("SECTION 7: RUN ACTUAL ACCIDENT VIDEO THROUGH YOLO11-AMF ONLY")
    print("=" * 60)

    cap = cv2.VideoCapture(str(target_video_path))
    frame_idx = 0
    stride = 1  # Full dense frame-by-frame processing for audit
    processed_count = 0
    detections_log = []
    frames_with_detections = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        timestamp = frame_idx / fps
        res = model(frame, conf=0.15, iou=0.45, verbose=False)[0]
        preds = []
        for b in res.boxes:
            conf = float(b.conf[0].cpu().numpy())
            cls_id = int(b.cls[0].cpu().numpy())
            xyxy = [round(float(c), 1) for c in b.xyxy[0].cpu().numpy()]
            preds.append({"class_id": cls_id, "conf": round(conf, 4), "box": xyxy})

        if preds:
            frames_with_detections += 1
            max_c = max([p["conf"] for p in preds])
        else:
            max_c = 0.0

        detections_log.append({
            "frame_idx": frame_idx,
            "timestamp": round(timestamp, 3),
            "detections": preds,
            "max_conf": max_c
        })
        processed_count += 1
        frame_idx += 1

    cap.release()

    print(f"Total video frames processed: {processed_count}")
    print(f"Frames with YOLO detections (conf >= 0.15): {frames_with_detections}")

    # Inspect window around ground truth accident frame (frame 109, range 95-135)
    print("\n--- Diagnostic Report for Accident Window (Frames 95 to 135) ---")
    window_dets = [d for d in detections_log if 95 <= d["frame_idx"] <= 135]
    accident_seen = False
    for wd in window_dets:
        f_num = wd["frame_idx"]
        t_sec = wd["timestamp"]
        m_c = wd["max_conf"]
        d_list = wd["detections"]
        if d_list:
            accident_seen = True
            for d in d_list:
                print(f"Frame {f_num:3d} ({t_sec:5.2f}s) | Detection: Accident | Conf: {d['conf']*100:5.1f}% | Box: {d['box']}")
        else:
            if f_num % 5 == 0 or f_num == accident_frame_gt:
                print(f"Frame {f_num:3d} ({t_sec:5.2f}s) | Detection: None     | Conf:   0.0% | Box: None")

    print(f"\nAccident Window Detection Result: {'A. YOLO SEES THE ACCIDENT' if accident_seen else 'B. YOLO MISSES THE ACCIDENT'}")

if __name__ == "__main__":
    run_audit()
