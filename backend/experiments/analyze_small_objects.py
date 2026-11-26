"""
Experiment 2: Small Object Bounding Box Size Distribution Analysis
Analyzes training annotations in dataset/prepared/labels/train/ and original image resolutions.
Categorizes boxes into:
- <16 px
- 16–32 px
- 32–64 px
- 64–128 px
- >128 px
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
import json

WORKSPACE = Path(__file__).resolve().parent.parent.parent
TRAIN_IMG_DIR = WORKSPACE / "dataset" / "prepared" / "images" / "train"
TRAIN_LBL_DIR = WORKSPACE / "dataset" / "prepared" / "labels" / "train"

print("=" * 70)
print("EXPERIMENT 2: SMALL OBJECT ANALYSIS (TRAINING ANNOTATIONS)")
print("=" * 70)

# We will calculate at both:
# 1) The image's native resolution
# 2) Standard 640x640 resolution

groups = {
    "<16 px": 0,
    "16-32 px": 0,
    "32-64 px": 0,
    "64-128 px": 0,
    ">128 px": 0
}

groups_640 = {
    "<16 px": 0,
    "16-32 px": 0,
    "32-64 px": 0,
    "64-128 px": 0,
    ">128 px": 0
}

total_objects = 0
empty_label_files = 0
non_empty_label_files = 0
all_max_dims = []
all_areas = []

lbl_files = sorted(list(TRAIN_LBL_DIR.glob("*.txt")))
print(f"Total training label files: {len(lbl_files)}")

for lbl_path in lbl_files:
    img_path = TRAIN_IMG_DIR / f"{lbl_path.stem}.jpg"
    if not img_path.exists():
        # try png
        img_path = TRAIN_IMG_DIR / f"{lbl_path.stem}.png"
    
    # Read image resolution
    if img_path.exists():
        with Image.open(img_path) as im:
            img_w, img_h = im.size
    else:
        img_w, img_h = 640, 640

    with open(lbl_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    if not lines:
        empty_label_files += 1
        continue
    
    non_empty_label_files += 1
    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            norm_w = float(parts[3])
            norm_h = float(parts[4])

            # Dimension at native image size: max(w_px, h_px)
            w_px_native = norm_w * img_w
            h_px_native = norm_h * img_h
            max_dim_native = max(w_px_native, h_px_native)

            # Dimension at 640x640
            w_px_640 = norm_w * 640
            h_px_640 = norm_h * 640
            max_dim_640 = max(w_px_640, h_px_640)

            total_objects += 1
            all_max_dims.append(max_dim_native)
            all_areas.append(w_px_native * h_px_native)

            # Grouping by max dimension (COCO definition uses area, user asks by px bounds: <16, 16-32, 32-64, 64-128, >128)
            if max_dim_native < 16:
                groups["<16 px"] += 1
            elif max_dim_native < 32:
                groups["16-32 px"] += 1
            elif max_dim_native < 64:
                groups["32-64 px"] += 1
            elif max_dim_native < 128:
                groups["64-128 px"] += 1
            else:
                groups[">128 px"] += 1

            if max_dim_640 < 16:
                groups_640["<16 px"] += 1
            elif max_dim_640 < 32:
                groups_640["16-32 px"] += 1
            elif max_dim_640 < 64:
                groups_640["32-64 px"] += 1
            elif max_dim_640 < 128:
                groups_640["64-128 px"] += 1
            else:
                groups_640[">128 px"] += 1

print(f"Total Objects in Training Set: {total_objects}")
print(f"Label files with annotations: {non_empty_label_files}")
print(f"Label files without annotations (negative frames): {empty_label_files}")

print("\n--- SIZE DISTRIBUTION (AT NATIVE IMAGE RESOLUTION) ---")
print(f"{'Size Category':<15} | {'Object Count':<14} | {'Percentage':<12}")
print("-" * 45)
for cat, cnt in groups.items():
    pct = (cnt / total_objects * 100) if total_objects > 0 else 0
    print(f"{cat:<15} | {cnt:<14} | {pct:<11.2f}%")

sub32_native = groups["<16 px"] + groups["16-32 px"]
sub32_native_pct = (sub32_native / total_objects * 100) if total_objects > 0 else 0
print(f"\nTotal accident objects < 32 px at native resolution: {sub32_native} ({sub32_native_pct:.2f}%)")

print("\n--- SIZE DISTRIBUTION (AT 640x640 INFERENCE RESOLUTION) ---")
print(f"{'Size Category':<15} | {'Object Count':<14} | {'Percentage':<12}")
print("-" * 45)
for cat, cnt in groups_640.items():
    pct = (cnt / total_objects * 100) if total_objects > 0 else 0
    print(f"{cat:<15} | {cnt:<14} | {pct:<11.2f}%")

sub32_640 = groups_640["<16 px"] + groups_640["16-32 px"]
sub32_640_pct = (sub32_640 / total_objects * 100) if total_objects > 0 else 0
print(f"\nTotal accident objects < 32 px at 640x640 resolution: {sub32_640} ({sub32_640_pct:.2f}%)")

report = {
    "total_objects": total_objects,
    "native_distribution": groups,
    "native_sub_32px_count": sub32_native,
    "native_sub_32px_pct": round(sub32_native_pct, 2),
    "resolution_640_distribution": groups_640,
    "resolution_640_sub_32px_count": sub32_640,
    "resolution_640_sub_32px_pct": round(sub32_640_pct, 2)
}

out_p = WORKSPACE / "results" / "metrics" / "small_object_analysis.json"
out_p.parent.mkdir(parents=True, exist_ok=True)
with open(out_p, "w") as f:
    json.dump(report, f, indent=2)

print(f"\nAnalysis saved to {out_p}")
