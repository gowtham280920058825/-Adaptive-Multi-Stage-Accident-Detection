# Dataset Documentation & Reproducibility Guide

This directory contains the dataset metadata, prepared YOLO train/validation/test splits, evaluation clips, and representative real and synthetic video samples used by the **Adaptive Multi-Stage Framework for Vision-Based Road Accident Detection**.

> **Note on Repository Size**: The complete raw dataset on disk is **~19.7 GB** (2,027 real videos and 2,211 synthetic CARLA videos + JSON annotations). To keep the GitHub repository cloneable and within GitHub hosting limits without breaking any pipeline, training, evaluation, or test scripts, bulk raw videos are excluded via `.gitignore`, while all metadata, prepared splits, evaluation videos, and representative samples are tracked in Git.

---

## 1. Included Files in This Repository

### A. Metadata, Class Schemas & Analysis Reports (100% Included)
| Path | Description |
| :--- | :--- |
| `metadata-real.csv` | Complete metadata for **2,027** real-world traffic accident videos (collision type, rollover flag, `accident_frame`, bounding boxes, weather, day/night, scene layout, official `split_in_distribution` & `split_geo_aware` splits). |
| `metadata-synthetic.csv` | Complete metadata for **2,211** synthetic CARLA traffic accident simulations (map, weather, camera position, collision type, bounding boxes). |
| `annotation_classes.yaml` | Semantic class definitions used in the synthetic annotations. |
| `dataset_analysis/DATASET_ANALYSIS.md` | Statistical breakdown of collision types, weather, lighting, scene layouts, and sample bounding-box verification results. |

### B. Prepared YOLO Dataset Splits (100% Included)
Located in `dataset/prepared/`, preserving the zero-leakage video-level split structure:
- `dataset/prepared/dataset.yaml` — Portable YOLO dataset configuration (`nc: 1`, class `0: Accident`).
- `dataset/prepared/images/train/` & `labels/train/` — **1,680** images + **1,680** YOLO annotations (`1,260` accident, `420` background).
- `dataset/prepared/images/val/` & `labels/val/` — **348** images + **348** YOLO annotations (`261` accident, `87` background).
- `dataset/prepared/images/test/` & `labels/test/` — **400** images + **400** YOLO annotations (`300` accident, `100` background), exact 1:1 pairing.
- `dataset/prepared/unmatched_test_samples/` — **12** leftover accident frames (`2wUrP2wgXhs_00_f42..f67_acc.jpg` and `JZcTEwl1Od4_00_f185..f210_acc.jpg`) from an earlier pilot extraction moved out of `images/test/` so unlabeled accident frames are not miscounted as background negatives during YOLO evaluation.
- *(Auto-generated `labels/*.cache` binary files are excluded and rebuilt automatically by Ultralytics.)*

### C. End-to-End Evaluation & Benchmark Videos (100% Included)
- `dataset/test_normal_traffic.mp4` — Normal highway traffic baseline clip.
- `dataset/evaluation_videos/accident/` (7 videos):
  `-6SQSDj8cYU_00.mp4`, `0isplBZQftg_00.mp4`, `0NixrF6Mt00_0_00.mp4`, `fzWY0vLAXzI_00.mp4`, `nAXTthLfgtI_00.mp4`, `Z4kg2Ev3vhk_00.mp4`, `__WFqm4i3vE_00.mp4`
- `dataset/evaluation_videos/non_accident/` (3 videos):
  `normal_traffic_01.mp4`, `normal_traffic_clear.mp4`, `normal_traffic_urban.mp4`

### D. Representative Real Videos (`dataset/real_videos/`, 19 Samples)
Selected to cover all **5 collision types** (`head-on`, `rear-end`, `sideswipe`, `single`, `t-bone`), day/night and weather variations, all automated tests (`tests/test_end_to_end_detection.py`), screenshot capture scripts, and every sample referenced in `DATASET_ANALYSIS.md`:
- **Head-on**: `SBTD7z4A1y8_00.mp4`, `6hJ8mYLUfE8_00.mp4`
- **Rear-end**: `CkI203kbHQ8_11_00.mp4`, `XPErIB4xgYg_00.mp4`, `Z4kg2Ev3vhk_00.mp4`
- **Sideswipe**: `2wUrP2wgXhs_00.mp4`, `7CO_LqgfiFg_00.mp4`
- **Single-vehicle**: `UcJKeuDYOzc_12_00.mp4`, `aERiIg9Dts4_00.mp4`, `nAXTthLfgtI_00.mp4`
- **T-bone**: `russia13_00.mp4`, `4Bde0lYQ4mg_00.mp4`, `fzWY0vLAXzI_00.mp4`, `-6SQSDj8cYU_00.mp4`
- **Additional Test & Diagnostic Cases**: `_9tp71S2Gwg_00.mp4`, `__WFqm4i3vE_00.mp4`, `_0RDZgCKPGo_00.mp4`, `0NixrF6Mt00_0_00.mp4`, `0isplBZQftg_00.mp4`

### E. Representative Synthetic Samples (`dataset/synthetic_videos/`, 5 Samples)
One video + matching frame-level JSON annotation per collision category (including `Town05_sideswipe_clear_00.mp4` used by `backend/training/evaluate_event_level.py`):
- `videos/head-on/Town03_head-on_clear_00.mp4` + `annotations/Town03_head-on_clear_00.json/`
- `videos/rear-end/Town04_rear-end_clear_00.mp4` + `annotations/Town04_rear-end_clear_00.json/`
- `videos/sideswipe/Town05_sideswipe_clear_00.mp4` + `annotations/Town05_sideswipe_clear_00.json/`
- `videos/single/Town07_single_clear_00.mp4` + `annotations/Town07_single_clear_00.json/`
- `videos/t-bone/Town03_t-bone_clear_00.mp4` + `annotations/Town03_t-bone_clear_00.json/`

---

## 2. Excluded Files (Preserved Locally, Ignored on GitHub)

- The remaining **2,008** bulk clips in `dataset/real_videos/` (~4.65 GB).
- The remaining **2,206** bulk CARLA videos and JSON annotation directories in `dataset/synthetic_videos/` (~14.5 GB).

---

## 3. How to Obtain & Restore the Remaining Original Dataset

1. **Source Dataset**: The raw dataset combines the **Nexar / YouTube Real-World Traffic Accident & CARLA Synthetic Accident Benchmark** (`archive.zip`, ~16.7 GB compressed), structured with `metadata-real.csv`, `metadata-synthetic.csv`, `real_videos/`, and `synthetic_videos/`.
   - Each row in `metadata-real.csv` records the relative `path` (e.g., `real_videos/<youtube_id>_00.mp4`), exact collision timestamp/frame, bounding box, and official train/test split.
   - Each row in `metadata-synthetic.csv` records the CARLA `rgb_path` (`synthetic_videos/videos/<type>/<town>_<type>_<weather>_<id>.mp4`) and corresponding `annotations_path`.
2. **Restoring Locally**:
   - Download or extract the original dataset archive into `dataset/` so that `dataset/real_videos/` and `dataset/synthetic_videos/` contain the full set of `.mp4` and `.json` files matching `metadata-real.csv` and `metadata-synthetic.csv`.
3. **Re-running Dataset Preparation & Verification** (optional, since `dataset/prepared/` is already included):
   ```bash
   python backend/training/dataset_inspector.py
   python backend/training/prepare_full_dataset.py
   ```
   Or for the lightweight 60-video stratified subset:
   ```bash
   python backend/training/prepare_dataset.py
   ```
