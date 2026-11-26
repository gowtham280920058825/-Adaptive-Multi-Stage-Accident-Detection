# Adaptive Multi-Stage Framework for Vision-Based Road Accident Detection and Intelligent Emergency Alert System

**M.Sc. R&D Project & Production Prototype**  

---

## 1. Research Objective & System Overview

This project implements a complete, end-to-end research prototype and customer-facing application for automated road accident detection, temporal event verification, severity classification, and intelligent emergency response routing.

### Proposed Research Architecture
```
VIDEO INPUT 
   │
   ▼
FRAME / TEMPORAL PROCESSING (Sliding Window Buffer)
   │
   ▼
BASE ACCIDENT DETECTOR (YOLO11 Baseline)
   │
   ▼
YOLO11-AMF (Base Paper Architecture: MLLA + AFPN + Focaler-IoU)
   │
   ▼
AMAVE (Adaptive Multi-Stage Accident Verification Engine - Proposed Module 1)
   │
   ▼
SASA (Smart Accident Severity Analyzer - Proposed Module 2)
   │
   ▼
SEDE (Smart Emergency Decision Engine - Proposed Module 3)
   │
   ▼
FINAL VERIFIED ACCIDENT RESULT & EMERGENCY RESPONSE DECISION
   │
   ▼
PROFESSIONAL CUSTOMER-FACING DASHBOARD & PDF/JSON REPORT EXPORT
```

---

## 2. Primary Research Foundation (Base Paper)

- **Title**: A Deep Learning Framework for Traffic Accident Detection Based on Improved YOLO11
- **Authors**: Weijun Li, Liyan Huang, Xiaofeng Lai
- **Journal**: *Vehicles*, 2025
- **Base Paper Contributions**: Improved YOLO11 architecture integrating:
  1. **MLLA** (Multi-head Linear Local Attention) for contextual feature enhancement.
  2. **AFPN** (Asymptotic Feature Pyramid Network) for multi-scale feature fusion.
  3. **Focaler-IoU** loss function for hard example localization.

---

## 3. Proposed Research Contributions (My Work)

1. **AMAVE (Adaptive Multi-Stage Accident Verification Engine)**:
   A 5-stage temporal verification engine evaluating detection confidence, multi-frame temporal persistence, motion vector consistency, and spatial overlap. Effectively filters false alarms from sudden braking, traffic congestion, brief occlusion, and camera shake.

2. **SASA (Smart Accident Severity Analyzer)**:
   A multi-feature severity classification model running only after AMAVE confirmation. Assesses involved vehicles, impact area proxies, collision geometry, and corridor layout to output severity levels (`LOW`, `MODERATE`, `HIGH`, `CRITICAL`).

3. **SEDE (Smart Emergency Decision Engine)**:
   A transparent policy engine issuing simulated emergency response recommendations (control room alerts, medical dispatch, traffic police routing).

---

## 4. System Requirements & Device Capabilities

- **Python**: 3.10+
- **PyTorch**: 2.0+ (Automatic CUDA GPU detection with CPU fallback)
- **Dependencies**: `ultralytics`, `opencv-python`, `fastapi`, `uvicorn`, `pandas`, `numpy`, `pillow`, `matplotlib`

```
CUDA available: False (Fallback to CPU processing)
Active Model: YOLO11-AMF
Database: NONE (Local file persistence under results/)
Authentication: NONE (Opens directly to main dashboard)
```

---

## 5. Directory Layout

```
d:\New folder\r&d\
├── frontend/             # HTML5/CSS3/JS Customer Dashboard
├── backend/              # FastAPI Server, Detection, AMAVE, SASA, SEDE
├── dataset/              # Dataset metadata, prepared YOLO splits, evaluation clips & samples (see dataset/README.md)
├── results/              # Detections, annotated videos, JSON reports, metrics, plots
├── experiments/          # Baseline, base_paper, and proposed ablation study results
├── tests/                # Pytest unit & integration tests
├── run_backend.bat       # Windows FastAPI server launcher
├── run_frontend.bat      # Windows Dashboard browser launcher
├── train_gpu.bat         # Model training script
└── evaluate.bat          # 5-Stage Ablation Study benchmark script
```

---

## 6. Quick Start & Execution

### Running the Customer Dashboard
1. Start the FastAPI backend server:
   ```cmd
   run_backend.bat
   ```
2. Launch the frontend dashboard:
   ```cmd
   run_frontend.bat
   ```
3. Open `http://127.0.0.1:8000` in your browser. Upload a traffic video from `d:\New folder\dataset\real_videos\` and click **Analyze Accident Pipeline**.

---

## 7. Model Training & Evaluation

To execute model training:
```cmd
train_gpu.bat
```

To run the complete 5-stage Ablation Benchmark:
```cmd
evaluate.bat
```

---

## 8. Research Limitations & Future Work

- Bounding box annotations are aligned to accident impact frames in the dataset.
- Real emergency calls are simulated for customer safety.
