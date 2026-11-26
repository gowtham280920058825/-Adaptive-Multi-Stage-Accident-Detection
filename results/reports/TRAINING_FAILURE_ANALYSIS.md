# COMPREHENSIVE TRAINING FAILURE & PIPELINE AUDIT REPORT

**Date**: August 2026  
**Audited Model**: Base Paper YOLO11-AMF (`experiments/base_paper/best.pt`)  
**Evaluation Performance**: Precision: 0.2384 | Recall: 0.1295 | mAP50: 0.0811 | mAP50-95: 0.0392 | F1: 0.1678  

---

## A. Root Causes of Low Initial Performance

1. **Under-Training / Insufficient Epoch Count (Primary Driver)**:
   - The model was trained for **only 3 epochs** on 552 training images (35 batches per epoch = **105 total SGD steps**).
   - In 105 steps, the newly initialized single-class detection head (`nc=1`) only completed warm-up learning rates and initial loss stabilization. It had not yet converged to optimize precision or recall.

2. **Single-Frame Temporal Snapshot Limitations**:
   - The current formulation extracts **1 single frame at `accident_frame`** and **1 background frame prior to accident**.
   - Many real traffic videos contain motion blur, severe camera shake, or occlusions at the exact collision frame, making single-frame static object detection challenging without temporal feature accumulation.

3. **Sample Size & Sub-Set Distribution**:
   - Training was conducted on a **400-video subset** out of the 2,027 total available real videos. 
   - 400 videos yield only **552 training images**, **160 validation images**, and **682 test evaluation frames**. While sufficient for quick pipeline validation, 552 frames is small for high mAP object detection without heavy augmentation or 30-50 epochs.

---

## B. Dataset & Split Audit

| Split | Images | Label Files | Positive Accident | Background Normal | Disjoint Split Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 552 | 552 | 276 | 276 | DISJOINT |
| **Val** | 160 | 160 | 80 | 80 | DISJOINT |
| **Test** | 682 | 682 | 341 | 341 | DISJOINT |

- **Data Leakage Check**: **0 video overlap** between train, val, and test splits. Videos are strictly partitioned by video ID.
- **Image/Label Alignment**: 100% 1-to-1 matching between `.jpg` image files and `.txt` label files across all splits.

---

## C. Annotation & Label Verification

- **Label Format**: 100% compliant with standard YOLO format (`0 center_x center_y width height`).
- **Coordinate Bounds**: All coordinates (`cx, cy, w, h`) are normalized in the range `[0.0, 1.0]`.
- **Validation Errors**: **0 malformed label files** detected out of 697 ground-truth bounding boxes.
- **Bounding Box Quality**: Rendered 10 random training samples under `results/reports/dataset_sample_verification/`. Boxes accurately surround the physical vehicle collision impact region.

---

## D. Model Architecture & Component Verification

- **YOLO11-AMF Base Architecture**: Pretrained weights (`yolov8n.pt`) were loaded successfully, transferring **319 / 355 layers**.
- **MLLA (Multi-head Linear Local Attention)**: Initialized with 4 attention heads and 128 channel dimension.
- **AFPN (Asymptotic Feature Pyramid Network)**: Initialized multi-scale feature fusion across 64, 128, and 256 channel depths.
- **Focaler-IoU**: Configured with focal scaling parameter `d=0.5` and IoU threshold `0.45`.

---

## E. Training Configuration Audit

- **Epochs**: 3 (Insufficient for convergence)
- **Batch Size**: 16
- **Image Size**: 480x480
- **Optimizer**: AdamW (lr=0.002, momentum=0.9)
- **Device**: PyTorch execution on CPU/GPU fallback engine
- **Evaluation Weight Integrity**: Verified `experiments/base_paper/best.pt` was correctly loaded during evaluation.

---

## F. Corrective Action Plan & Next Training Recommendations

1. **Increase Epochs to 30–50**:
   - 30 epochs on 552 images = **1,050 training steps**, allowing learning rates to decay properly and head weights to converge.

2. **Expand Training Subset**:
   - Increase `max_samples` from 400 to **1,200+ videos** (yielding ~2,000 training images) for richer scene diversity across weather, lighting, and layout categories.

3. **Data Augmentation Tuning**:
   - Enable scale, mosaic, and HSV color jittering to improve generalization to difficult weather and night-time lighting conditions.

4. **Multi-Frame Sampling per Video**:
   - Extract 3 frames around the collision moment (`accident_frame - 5`, `accident_frame`, `accident_frame + 5`) to double the positive training samples per video without false annotations.

---

## G. Summary Audit Matrix

| Audit Check | Status | Finding |
| :--- | :--- | :--- |
| **YOLO Label Format** | VALID | 100% compliant (`0 cx cy w h`) |
| **Coordinate Normalization** | VALID | All values strictly within `[0.0, 1.0]` |
| **Train/Val/Test Leakage** | NONE | 0 video overlap across splits |
| **Pretrained Weights Loaded** | YES | Transferred 319/355 layers from backbone |
| **Bounding Box Alignment** | VERIFIED | GT boxes correctly bound collision regions |
| **Primary Root Cause** | IDENTIFIED | Under-training (3 epochs = 105 iterations) |
