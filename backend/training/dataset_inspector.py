"""
Dataset Inspector & Verification Engine
Performs comprehensive dataset inspection, verifies bounding box annotations on small samples,
renders sample images/videos, and generates DATASET_ANALYSIS.md report.
"""

import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure backend path is accessible
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from backend.config import DATASET_RAW_DIR, BASE_DIR, RESULTS_DIR
from backend.utils.logging_utils import get_logger

logger = get_logger("DatasetInspector")

def inspect_dataset() -> dict:
    """Analyze real and synthetic dataset metadata and verify video files."""
    real_csv = DATASET_RAW_DIR / "metadata-real.csv"
    synth_csv = DATASET_RAW_DIR / "metadata-synthetic.csv"
    real_video_dir = DATASET_RAW_DIR / "real_videos"

    if not real_csv.exists():
        raise FileNotFoundError(f"Metadata file not found: {real_csv}")

    df_real = pd.read_csv(real_csv)
    df_synth = pd.read_csv(synth_csv) if synth_csv.exists() else pd.DataFrame()

    total_real_videos = len(list(real_video_dir.glob("*.mp4"))) if real_video_dir.exists() else 0

    stats = {
        "total_real_csv_entries": len(df_real),
        "total_real_video_files": total_real_videos,
        "total_synth_csv_entries": len(df_synth),
        "collision_types": df_real["type"].value_counts().to_dict(),
        "rollover_counts": df_real["rollover"].value_counts().to_dict(),
        "weather_distribution": df_real["weather"].value_counts().to_dict(),
        "daytime_distribution": df_real["day_time"].value_counts().to_dict(),
        "scene_layouts": df_real["scene_layout"].value_counts().to_dict(),
        "quality_distribution": df_real["quality"].value_counts().to_dict(),
        "in_dist_split": df_real["split_in_distribution"].value_counts().to_dict(),
        "geo_aware_split": df_real["split_geo_aware"].value_counts().to_dict(),
        "avg_duration": float(df_real["duration"].mean()),
        "avg_frames": float(df_real["no_frames"].mean()),
    }
    return stats, df_real

def verify_sample_annotations(df_real: pd.DataFrame, num_samples: int = 10) -> list:
    """Select sample videos, extract accident frame, draw ground-truth bounding box, save sample verification images."""
    sample_dir = RESULTS_DIR / "reports" / "dataset_sample_verification"
    sample_dir.mkdir(parents=True, exist_ok=True)

    verified_samples = []
    # Sample from each collision type
    sampled_df = df_real.groupby("type", group_keys=False).apply(lambda x: x.sample(min(len(x), max(1, num_samples // 5)), random_state=42))

    for idx, row in sampled_df.iterrows():
        rel_path = row["path"]
        video_path = DATASET_RAW_DIR / rel_path
        if not video_path.exists():
            continue

        cap = cv2.VideoCapture(str(video_path))
        target_frame_idx = int(row["accident_frame"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            continue

        h, w, _ = frame.shape
        x1 = int(row["x1"] * w)
        y1 = int(row["y1"] * h)
        x2 = int(row["x2"] * w)
        y2 = int(row["y2"] * h)

        # Draw box on frame
        color = (0, 0, 255)  # Red for accident box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label_text = f"ACCIDENT: {row['type']} (Frame {target_frame_idx})"
        cv2.putText(frame, label_text, (max(10, x1), max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        out_img_name = f"sample_{Path(rel_path).stem}_frame{target_frame_idx}.jpg"
        out_img_path = sample_dir / out_img_name
        cv2.imwrite(str(out_img_path), frame)

        verified_samples.append({
            "video": rel_path,
            "accident_frame": target_frame_idx,
            "accident_type": row["type"],
            "box": [x1, y1, x2, y2],
            "sample_image": str(out_img_path)
        })

    logger.info(f"Verified {len(verified_samples)} sample accident frames. Saved to {sample_dir}")
    return verified_samples

def generate_dataset_report(stats: dict, samples: list):
    """Generate dataset/dataset_analysis/DATASET_ANALYSIS.md."""
    out_dir = BASE_DIR / "dataset" / "dataset_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "DATASET_ANALYSIS.md"

    md_content = f"""# DATASET ANALYSIS & VERIFICATION REPORT

**Dataset Location**: `d:\\New folder\\dataset`  
**Total Real Videos**: {stats['total_real_video_files']}  
**Total Real Metadata Records**: {stats['total_real_csv_entries']}  
**Total Synthetic Metadata Records**: {stats['total_synth_csv_entries']}  

---

## 1. Collision Category Distribution

| Collision Type | Count | Percentage |
| :--- | :--- | :--- |
"""
    total = stats['total_real_csv_entries']
    for k, v in stats['collision_types'].items():
        pct = (v / total) * 100
        md_content += f"| {k} | {v} | {pct:.2f}% |\n"

    md_content += f"""
---

## 2. Environmental & Context Features

### Weather Distribution
"""
    for k, v in stats['weather_distribution'].items():
        md_content += f"- **{k}**: {v} ({v/total*100:.1f}%)\n"

    md_content += "\n### Day / Night Distribution\n"
    for k, v in stats['daytime_distribution'].items():
        md_content += f"- **{k}**: {v} ({v/total*100:.1f}%)\n"

    md_content += "\n### Scene Layout Distribution\n"
    for k, v in stats['scene_layouts'].items():
        md_content += f"- **{k}**: {v} ({v/total*100:.1f}%)\n"

    md_content += f"""
---

## 3. Video Statistics

- **Average Video Duration**: {stats['avg_duration']:.2f} seconds
- **Average Frame Count**: {stats['avg_frames']:.1f} frames
- **In-Distribution Train/Test Split**: {stats['in_dist_split']}
- **Geo-Aware Train/Test Split**: {stats['geo_aware_split']}

---

## 4. Sample Bounding Box Verification Results

Verified **{len(samples)}** representative frames from real traffic videos. Sample rendered images saved in `results/reports/dataset_sample_verification/`.

| Video File | Accident Type | Target Frame | Rendered Sample |
| :--- | :--- | :--- | :--- |
"""
    for s in samples:
        img_name = Path(s['sample_image']).name
        md_content += f"| `{s['video']}` | {s['accident_type']} | {s['accident_frame']} | `{img_name}` |\n"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"Dataset analysis report generated at {report_file}")

if __name__ == "__main__":
    logger.info("Starting dataset inspection and sample verification...")
    stats, df_real = inspect_dataset()
    samples = verify_sample_annotations(df_real, num_samples=10)
    generate_dataset_report(stats, samples)
