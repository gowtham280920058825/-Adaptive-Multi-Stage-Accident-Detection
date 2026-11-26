"""
Phase 3 & Phase 4: Raw YOLO Diagnostic & Visualization
Target Video: _9tp71S2Gwg_00.mp4
Bypasses AMAVE, SASA, SEDE, UI completely.
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

WORKSPACE = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(WORKSPACE))

from backend.config import YOLO_AMF_WEIGHTS, DEVICE

VIDEO_PATH = WORKSPACE / "dataset" / "real_videos" / "_9tp71S2Gwg_00.mp4"
DEBUG_DIR = WORKSPACE / "debug" / "raw_yolo"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("PHASE 3: RAW YOLO TEST (EVERY FRAME INFERENCE)")
print("=" * 70)
print(f"Model: {YOLO_AMF_WEIGHTS}")
print(f"Video: {VIDEO_PATH}")

model = YOLO(str(YOLO_AMF_WEIGHTS))

cap = cv2.VideoCapture(str(VIDEO_PATH))
all_frames = []
idx = 0
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    all_frames.append((idx, frame))
    idx += 1
cap.release()

total_video_frames = len(all_frames)
print(f"Loaded {total_video_frames} frames into memory.")

# Run raw inference on every single frame with conf=0.01 (imgsz=640)
t0 = time.time()
raw_predictions = {}
max_raw_conf = 0.0
max_raw_frame = -1
max_raw_box = None
frames_with_any_detection = 0

print("Running YOLO11-AMF inference on every frame at conf=0.01...")
for f_idx, frame in all_frames:
    results = model(frame, conf=0.01, iou=0.40, imgsz=640, device=DEVICE, verbose=False)
    frame_boxes = []
    for r in results:
        for b in r.boxes:
            conf = float(b.conf[0].cpu().numpy())
            xyxy = [round(float(c), 2) for c in b.xyxy[0].cpu().numpy()]
            cls_id = int(b.cls[0].cpu().numpy())
            label = model.names.get(cls_id, "Accident")
            box_item = {
                "box": xyxy,
                "confidence": round(conf, 4),
                "class_id": cls_id,
                "label": label
            }
            frame_boxes.append(box_item)
            if conf > max_raw_conf:
                max_raw_conf = conf
                max_raw_frame = f_idx
                max_raw_box = box_item

    raw_predictions[f_idx] = frame_boxes
    if len(frame_boxes) > 0:
        frames_with_any_detection += 1

inf_time = time.time() - t0
print(f"Raw inference completed in {inf_time:.2f}s ({total_video_frames/inf_time:.1f} FPS)")
print(f"\n--- RAW YOLO METRICS SUMMARY ---")
print(f"MAXIMUM RAW YOLO CONFIDENCE: {max_raw_conf:.4f} ({max_raw_conf*100:.2f}%)")
print(f"FRAME WITH MAXIMUM CONFIDENCE: {max_raw_frame} (time: {max_raw_frame/14.646:.2f}s)")
print(f"NUMBER OF FRAMES WITH ANY DETECTION (conf >= 0.01): {frames_with_any_detection} / {total_video_frames} ({frames_with_any_detection/total_video_frames*100:.1f}%)")
if max_raw_box:
    print(f"Bounding Box at Max Frame: {max_raw_box['box']}, Class: {max_raw_box['label']} ({max_raw_box['class_id']})")

# Confidence sweep table
thresholds = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
sweep_table = []

print("\n" + "=" * 70)
print(f"{'Threshold':<10} | {'Detection Frames':<18} | {'Total Boxes':<14} | {'Max Confidence':<16}")
print("-" * 70)

for th in thresholds:
    det_frames = 0
    total_boxes = 0
    th_max_c = 0.0
    for f_idx, boxes in raw_predictions.items():
        matching = [b for b in boxes if b["confidence"] >= th]
        if matching:
            det_frames += 1
            total_boxes += len(matching)
            for m in matching:
                if m["confidence"] > th_max_c:
                    th_max_c = m["confidence"]
    sweep_table.append({
        "threshold": th,
        "detection_frames": det_frames,
        "total_boxes": total_boxes,
        "max_confidence": th_max_c
    })
    print(f"{th:<10.2f} | {det_frames:<18} | {total_boxes:<14} | {th_max_c:<16.4f}")

# Save JSON results
with open(DEBUG_DIR / "phase3_raw_yolo_sweep.json", "w") as f:
    json.dump({
        "video": str(VIDEO_PATH),
        "total_frames": total_video_frames,
        "max_raw_confidence": max_raw_conf,
        "max_raw_frame": max_raw_frame,
        "max_raw_box": max_raw_box,
        "frames_with_any_detection": frames_with_any_detection,
        "sweep": sweep_table
    }, f, indent=2)

print("\n" + "=" * 70)
print("PHASE 4: VISUALIZE YOLO DETECTIONS")
print("=" * 70)

# Ground truth accident frame is 83 from metadata
# 1. highest confidence frame
# 2. 10 frames before highest confidence (max_raw_frame - 10)
# 3. highest confidence frame (max_raw_frame)
# 4. 10 frames after highest confidence (max_raw_frame + 10)
# 5. several frames from the actual collision period around frame 83: 75, 80, 83, 85, 90

frames_to_visualize = sorted(list(set([
    max(0, max_raw_frame - 10),
    max(0, max_raw_frame - 5),
    max_raw_frame,
    min(total_video_frames - 1, max_raw_frame + 5),
    min(total_video_frames - 1, max_raw_frame + 10),
    75, 80, 83, 85, 90
])))

saved_images = []
frame_lookup = {f_idx: fr for f_idx, fr in all_frames}

for f_no in frames_to_visualize:
    base_frame = frame_lookup[f_no].copy()
    f_boxes = raw_predictions.get(f_no, [])
    # Draw detections
    for item in f_boxes:
        conf = item["confidence"]
        box = [int(v) for v in item["box"]]
        label = item["label"]
        color = (0, 255, 0) if conf >= 0.20 else (0, 165, 255)
        cv2.rectangle(base_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        text = f"ACCIDENT {conf*100:.1f}%"
        cv2.putText(base_frame, text, (max(10, box[0]), max(20, box[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(base_frame, text, (max(10, box[0]), max(20, box[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # Frame header
    header_text = f"Frame {f_no:03d} (Time {f_no/14.646:.2f}s) | Detections: {len(f_boxes)}"
    if f_no == max_raw_frame:
        header_text += " [PEAK CONFIDENCE FRAME]"
    if f_no == 83:
        header_text += " [GT ACCIDENT COLLISION FRAME]"

    cv2.rectangle(base_frame, (0, 0), (base_frame.shape[1], 45), (20, 20, 20), -1)
    cv2.putText(base_frame, header_text, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    img_filename = f"raw_yolo_frame_{f_no:03d}.jpg"
    img_path = DEBUG_DIR / img_filename
    cv2.imwrite(str(img_path), base_frame)
    saved_images.append(str(img_path))
    print(f"Saved: {img_path} ({len(f_boxes)} boxes, header: '{header_text}')")

print(f"\nPhase 4 complete. Saved {len(saved_images)} annotated diagnostic frames to debug/raw_yolo/")
