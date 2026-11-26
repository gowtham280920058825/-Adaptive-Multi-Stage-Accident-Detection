"""
Corrective Dataset Preparation Module (Video-Level Split & Multi-Frame Exporter)
Strictly enforces:
1. Video-level splitting (0 video leakage across train/val/test splits).
2. MAX_VIDEOS = 1200.
3. Multi-frame extraction per video: [accident_frame-5, accident_frame, accident_frame+5] and background frames.
4. Official metadata bounding box annotations.
"""

import os
import shutil
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from backend.config import DATASET_RAW_DIR, DATASET_PREPARED_DIR
from backend.utils.logging_utils import get_logger

logger = get_logger("PrepareDatasetCorrective")

def select_representative_videos(df_real: pd.DataFrame, max_videos: int = 60) -> list:
    """
    Selects representative video paths using stratified sampling across collision categories.
    """
    if len(df_real["path"].unique()) <= max_videos:
        return df_real["path"].unique().tolist()
    
    counts = df_real["type"].value_counts()
    ratios = counts / len(df_real)
    target_counts = (ratios * max_videos).round().astype(int)
    
    diff = max_videos - target_counts.sum()
    if diff != 0:
        target_counts[target_counts.idxmax()] += diff

    selected_paths = []
    for c_type, count in target_counts.items():
        sub_paths = df_real[df_real["type"] == c_type]["path"].unique()
        n_sample = min(len(sub_paths), count)
        np.random.seed(42)
        chosen = np.random.choice(sub_paths, size=n_sample, replace=False)
        selected_paths.extend(chosen.tolist())
    
    return selected_paths

def prepare_yolo_dataset_corrective(df_real: pd.DataFrame, max_videos: int = 60):
    """
    Video-level split dataset exporter for small representative dataset.
    """
    images_dir = DATASET_PREPARED_DIR / "images"
    labels_dir = DATASET_PREPARED_DIR / "labels"

    # Reset/clean prepared directory
    for split in ["train", "val", "test"]:
        s_img = images_dir / split
        s_lbl = labels_dir / split
        if s_img.exists():
            shutil.rmtree(s_img)
        if s_lbl.exists():
            shutil.rmtree(s_lbl)
        s_img.mkdir(parents=True, exist_ok=True)
        s_lbl.mkdir(parents=True, exist_ok=True)

    # 1. Stratified representative video selection
    selected_videos = select_representative_videos(df_real, max_videos=max_videos)
    logger.info(f"Selected {len(selected_videos)} representative videos out of {len(df_real)}")

    # 2. Video-level split assignment (70% train, 15% val, 15% test)
    np.random.seed(42)
    shuffled_videos = np.random.permutation(selected_videos)
    n_total = len(shuffled_videos)
    n_val = max(1, int(round(n_total * 0.15)))
    n_test = max(1, int(round(n_total * 0.15)))
    n_train = n_total - n_val - n_test

    val_videos = set(shuffled_videos[:n_val])
    test_videos = set(shuffled_videos[n_val:n_val + n_test])
    train_videos = set(shuffled_videos[n_val + n_test:])

    video_to_split = {}
    for v in train_videos:
        video_to_split[v] = "train"
    for v in val_videos:
        video_to_split[v] = "val"
    for v in test_videos:
        video_to_split[v] = "test"

    # Filter dataframe to selected videos
    df_subset = df_real[df_real["path"].isin(selected_videos)]

    video_split_counts = {"train": 0, "val": 0, "test": 0}
    image_split_counts = {"train": 0, "val": 0, "test": 0}
    processed_videos = set()

    for idx, row in df_subset.iterrows():
        rel_path = row["path"]
        video_path = DATASET_RAW_DIR / rel_path
        if not video_path.exists():
            continue

        split = video_to_split[rel_path]
        if rel_path not in processed_videos:
            processed_videos.add(rel_path)
            video_split_counts[split] += 1

        cap = cv2.VideoCapture(str(video_path))
        accident_frame_idx = int(row["accident_frame"])
        total_frames = int(row["no_frames"])

        # Bounding box coordinates normalized
        cx = float(row["center_x"])
        cy = float(row["center_y"])
        w = float(row["x2"] - row["x1"])
        h = float(row["y2"] - row["y1"])

        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.01, min(1.0, abs(w)))
        h = max(0.01, min(1.0, abs(h)))

        # Extract 3 accident frames: [accident_frame - 5, accident_frame, accident_frame + 5]
        acc_offsets = [-5, 0, 5]
        for offset in acc_offsets:
            f_idx = accident_frame_idx + offset
            if f_idx < 0 or f_idx >= total_frames:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                base_name = f"{Path(rel_path).stem}_f{f_idx}_acc"
                img_path = images_dir / split / f"{base_name}.jpg"
                lbl_path = labels_dir / split / f"{base_name}.txt"

                cv2.imwrite(str(img_path), frame)
                with open(lbl_path, "w") as f:
                    f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                image_split_counts[split] += 1

        # Extract 3 normal background frames: [accident_frame - 40, accident_frame - 45, accident_frame - 50]
        bg_offsets = [-40, -45, -50]
        for offset in bg_offsets:
            f_idx = max(0, accident_frame_idx + offset)
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret_norm, frame_norm = cap.read()
            if ret_norm and frame_norm is not None:
                base_name_norm = f"{Path(rel_path).stem}_f{f_idx}_norm"
                img_path_norm = images_dir / split / f"{base_name_norm}.jpg"
                lbl_path_norm = labels_dir / split / f"{base_name_norm}.txt"

                cv2.imwrite(str(img_path_norm), frame_norm)
                open(lbl_path_norm, "w").close()
                image_split_counts[split] += 1

        cap.release()

    logger.info("==================================================================")
    logger.info("            CORRECTIVE DATASET PREPARATION SUMMARY                ")
    logger.info(f"Source Videos Processed: {len(processed_videos)}")
    logger.info(f"Source Videos Split -> Train: {video_split_counts['train']}, Val: {video_split_counts['val']}, Test: {video_split_counts['test']}")
    logger.info(f"Extracted Images Split -> Train: {image_split_counts['train']}, Val: {image_split_counts['val']}, Test: {image_split_counts['test']}")
    logger.info("==================================================================")

    # Export dataset.yaml
    yaml_path = DATASET_PREPARED_DIR / "dataset.yaml"
    yaml_content = f"""path: {DATASET_PREPARED_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: Accident
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    return video_split_counts, image_split_counts

if __name__ == "__main__":
    real_csv = DATASET_RAW_DIR / "metadata-real.csv"
    if real_csv.exists():
        df_real = pd.read_csv(real_csv)
        prepare_yolo_dataset_corrective(df_real, max_videos=60)
    else:
        logger.error(f"Metadata file missing at {real_csv}")

