"""
Full Dataset Preparation Module (Video-Level Split & Multi-Frame Exporter)
Extracts the complete valid training dataset from dataset/metadata-real.csv:
- Train: 420 videos (~1,680 images: 1,260 accident + 420 background)
- Validation: 87 videos (~348 images: 261 accident + 87 background)
- Held-Out Test: 100 videos (~400 images: 300 accident + 100 background)
Strictly enforces:
1. Video-level separation (zero video leakage across splits).
2. Verified bounding box annotations in YOLO format (0 cx cy w h).
3. Background frames with empty labels for true negative training.
"""

import os
import shutil
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from backend.config import DATASET_RAW_DIR, DATASET_PREPARED_DIR
from backend.utils.logging_utils import get_logger

logger = get_logger("PrepareFullDataset")

def prepare_full_dataset():
    metadata_path = DATASET_RAW_DIR / "metadata-real.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    df = pd.read_csv(metadata_path)
    logger.info(f"Loaded {len(df)} total entries from {metadata_path.name}")

    # Official split_in_distribution
    train_pool = df[df["split_in_distribution"] == "train"].copy()
    test_pool = df[df["split_in_distribution"] == "test"].copy()

    logger.info(f"Official pool sizes: Train pool = {len(train_pool)} videos, Test pool = {len(test_pool)} videos")

    # Video-level split of train_pool into Train (420 videos) and Val (87 videos)
    np.random.seed(42)
    shuffled_train_pool = train_pool.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    n_train = 420
    n_val = len(shuffled_train_pool) - n_train  # 87 videos
    
    df_train = shuffled_train_pool.iloc[:n_train]
    df_val = shuffled_train_pool.iloc[n_train:]

    # Select 100 stratified test videos from test_pool (stratified by collision type)
    counts = test_pool["type"].value_counts()
    ratios = counts / len(test_pool)
    target_test_counts = (ratios * 100).round().astype(int)
    diff = 100 - target_test_counts.sum()
    if diff != 0:
        target_test_counts[target_test_counts.idxmax()] += diff

    test_selected = []
    for c_type, count in target_test_counts.items():
        sub = test_pool[test_pool["type"] == c_type]
        sampled = sub.sample(n=min(len(sub), count), random_state=42)
        test_selected.append(sampled)
    df_test = pd.concat(test_selected).reset_index(drop=True)

    logger.info(f"Final Video Split:")
    logger.info(f"  Train: {len(df_train)} unique videos")
    logger.info(f"  Validation: {len(df_val)} unique videos")
    logger.info(f"  Held-out Test: {len(df_test)} unique videos")

    # Verify zero overlap
    train_paths = set(df_train["path"])
    val_paths = set(df_val["path"])
    test_paths = set(df_test["path"])
    assert len(train_paths & val_paths) == 0, "Train and Val video overlap detected!"
    assert len(train_paths & test_paths) == 0, "Train and Test video overlap detected!"
    assert len(val_paths & test_paths) == 0, "Val and Test video overlap detected!"
    logger.info(" Verified: Zero video leakage across Train, Val, and Test splits.")

    images_dir = DATASET_PREPARED_DIR / "images"
    labels_dir = DATASET_PREPARED_DIR / "labels"

    # Reset prepared directories
    for s in ["train", "val", "test"]:
        s_img = images_dir / s
        s_lbl = labels_dir / s
        if s_img.exists():
            shutil.rmtree(s_img)
        if s_lbl.exists():
            shutil.rmtree(s_lbl)
        s_img.mkdir(parents=True, exist_ok=True)
        s_lbl.mkdir(parents=True, exist_ok=True)

    def extract_split(df_split, split_name):
        extracted_acc = 0
        extracted_bg = 0
        logger.info(f"Extracting frames for split '{split_name}' from {len(df_split)} videos...")

        for idx, row in df_split.iterrows():
            rel_path = row["path"]
            vid_path = DATASET_RAW_DIR / rel_path
            if not vid_path.exists():
                logger.warning(f"Video file missing: {vid_path}")
                continue

            cap = cv2.VideoCapture(str(vid_path))
            if not cap.isOpened():
                continue

            total_frames = int(row["no_frames"])
            acc_frame = int(row["accident_frame"])

            # Bounding box coordinates normalized
            cx = max(0.0, min(1.0, float(row["center_x"])))
            cy = max(0.0, min(1.0, float(row["center_y"])))
            w = max(0.01, min(1.0, abs(float(row["x2"] - row["x1"]))))
            h = max(0.01, min(1.0, abs(float(row["y2"] - row["y1"]))))

            v_stem = Path(rel_path).stem

            # Extract 3 accident frames: [acc_frame - 3, acc_frame, acc_frame + 3]
            acc_offsets = [-3, 0, 3]
            for offset in acc_offsets:
                f_idx = acc_frame + offset
                if 0 <= f_idx < total_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        fname = f"{v_stem}_f{f_idx}_acc"
                        img_out = images_dir / split_name / f"{fname}.jpg"
                        lbl_out = labels_dir / split_name / f"{fname}.txt"
                        cv2.imwrite(str(img_out), frame)
                        with open(lbl_out, "w") as f:
                            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                        extracted_acc += 1

            # Extract 1 background normal traffic frame (well before accident, e.g. acc_frame - 40)
            bg_idx = max(0, acc_frame - 40)
            cap.set(cv2.CAP_PROP_POS_FRAMES, bg_idx)
            ret_bg, frame_bg = cap.read()
            if ret_bg and frame_bg is not None:
                fname_bg = f"{v_stem}_f{bg_idx}_bg"
                img_out = images_dir / split_name / f"{fname_bg}.jpg"
                lbl_out = labels_dir / split_name / f"{fname_bg}.txt"
                cv2.imwrite(str(img_out), frame_bg)
                # Empty file for background image
                with open(lbl_out, "w") as f:
                    pass
                extracted_bg += 1

            cap.release()

        logger.info(f"Split '{split_name}' completed: {extracted_acc} accident frames, {extracted_bg} background frames. Total = {extracted_acc + extracted_bg}")
        return extracted_acc, extracted_bg

    extract_split(df_train, "train")
    extract_split(df_val, "val")
    extract_split(df_test, "test")

    # Update dataset.yaml
    prep_dir_str = str(DATASET_PREPARED_DIR).replace('\\', '/')
    yaml_content = f"""path: {prep_dir_str}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: Accident
"""
    yaml_file = DATASET_PREPARED_DIR / "dataset.yaml"
    with open(yaml_file, "w") as f:
        f.write(yaml_content)

    logger.info(f"Updated {yaml_file} successfully.")
    logger.info("Full dataset preparation complete.")

if __name__ == "__main__":
    prepare_full_dataset()
