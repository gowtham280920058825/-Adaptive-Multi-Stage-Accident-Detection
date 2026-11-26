"""
Dataset Validation & Sample Verification Script for Corrective Dataset
Renders 20 random samples across train/val/test splits under results/reports/dataset_sample_verification_corrective/
"""

import os
import random
import cv2
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from backend.config import DATASET_PREPARED_DIR, RESULTS_DIR

def verify_corrective_samples():
    out_dir = RESULTS_DIR / "reports" / "dataset_sample_verification_corrective"
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir = DATASET_PREPARED_DIR / "images"
    labels_dir = DATASET_PREPARED_DIR / "labels"

    samples_to_check = []
    for split in ["train", "val", "test"]:
        # Find positive accident images (ending in _acc.jpg)
        acc_imgs = list((images_dir / split).glob("*_acc.jpg"))
        count = 10 if split == "train" else 5
        selected = random.sample(acc_imgs, min(count, len(acc_imgs)))
        for img_p in selected:
            samples_to_check.append((split, img_p))

    verified_meta = []
    print(f"Verifying {len(samples_to_check)} dataset samples...")

    for s_idx, (split, img_path) in enumerate(samples_to_check):
        lbl_path = labels_dir / split / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None or not lbl_path.exists():
            continue

        h_img, w_img, _ = img.shape
        with open(lbl_path, "r") as f:
            lines = f.readlines()

        box_drawn = False
        cls_id = None
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
                x1 = int((cx - bw / 2) * w_img)
                y1 = int((cy - bh / 2) * h_img)
                x2 = int((cx + bw / 2) * w_img)
                y2 = int((cy + bh / 2) * h_img)

                # Draw green box for verified bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                tag_text = f"Split: {split} | Class: {cls_id} (Accident)"
                cv2.putText(img, tag_text, (max(10, x1), max(25, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                box_drawn = True

        file_text = f"File: {img_path.name}"
        cv2.putText(img, file_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out_name = f"verify_{split}_sample_{s_idx+1}_{img_path.name}"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), img)

        verified_meta.append({
            "split": split,
            "filename": img_path.name,
            "class_id": cls_id,
            "has_valid_box": box_drawn,
            "saved_image": str(out_path)
        })

    print(f"Successfully rendered and verified {len(verified_meta)} sample images under {out_dir}")
    return verified_meta

if __name__ == "__main__":
    verify_corrective_samples()
