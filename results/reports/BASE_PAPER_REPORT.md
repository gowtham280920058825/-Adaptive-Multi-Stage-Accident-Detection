# BASE-PAPER (YOLO11-AMF) EVALUATION & COMPARISON REPORT

**Model Weight Path**: `experiments/base_paper/best.pt`  
**Base Architecture**: YOLO11 (`yolo11n.pt`)  
**Integrated Components**: MLLA (Multi-head Linear Local Attention), AFPN (Asymptotic Feature Pyramid Network), Focaler-IoU  
**Evaluation Target**: Exact same 54 test images from `dataset/prepared` (9 test videos, 0 video leakage)  
**GPU Device**: NVIDIA GeForce RTX 2050  

---

## 1. Direct Comparison: Baseline YOLO11n vs Base-Paper YOLO11-AMF

| Metric | Baseline YOLO11n | Base-Paper YOLO11-AMF | Change |
| :--- | :--- | :--- | :--- |
| **Precision** | 0.3123 | 0.3730 | +0.0607 |
| **Recall** | 0.0741 | 0.0741 | +0.0000 |
| **mAP@50** | 0.0626 | 0.0631 | +0.0005 |
| **mAP@50-95** | 0.0184 | 0.0186 | +0.0001 |
| **F1-Score** | 0.1197 | 0.1236 | +0.0039 |
| **Inference Speed** | ~4.80 ms/img | 8.47 ms/img (118.1 FPS) | - |

---

## 2. Dataset & Split Specifications

- **Videos Used**: 60 (Stratified real accident videos)
- **Train Split**: 42 videos (241 images)
- **Validation Split**: 9 videos (48 images)
- **Test Split**: 9 videos (54 images)
- **Data Leakage**: 0 videos overlapping across splits

---

## 3. Training & Hardware Configuration

- **GPU Hardware**: NVIDIA GeForce RTX 2050 (4 GB VRAM / CUDA 12.1 engine)
- **Batch Size**: 8
- **Image Size**: 480 × 480
- **Epochs**: 20
- **Focaler-IoU Alignment**: `iou=0.45`
