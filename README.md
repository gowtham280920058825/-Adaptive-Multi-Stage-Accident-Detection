# Adaptive Multi-Stage Framework for Vision-Based Road Accident Detection and Intelligent Emergency Alert System

**M.Sc. R&D Project & End-to-End Decision-Support Prototype**

---

## 1. Project Introduction & Research Objectives

Single-stage object detectors can localize candidate road crashes in individual frames, but in real-world traffic surveillance they frequently generate false alarms during sudden braking, dense congestion, brief occlusions, and camera vibration.

This repository implements an end-to-end research prototype and web application that pairs YOLO11 candidate accident detection with three proposed post-detection verification and decision-support engines:
1. **Temporal False-Alarm Suppression (`AMAVE V2`)** — Verify candidate detections across consecutive frames using trajectory persistence, Farneback optical flow motion consistency, and spatial vehicle interaction.
2. **Evidence-Grounded Severity Estimation (`SASA`)** — Quantify visual scene severity (`LOW`, `MODERATE`, `HIGH`, `CRITICAL`) strictly after AMAVE confirmation.
3. **Policy-Driven Emergency Decision Support (`SEDE`)** — Produce transparent, auditable control-room and emergency dispatch recommendations without placing automated external calls.

---

## 2. System Architecture

```text
VIDEO INPUT (MP4 / AVI / MOV, Stride = 2)
   │
   ▼
FRAME / TEMPORAL BUFFER (Sliding Window: 10 frames)
   │
   ▼
CANDIDATE ACCIDENT DETECTOR (YOLO11-AMF / YOLO11 Baseline)
   │
   ▼
AMAVE V2 (Adaptive Multi-Stage Accident Verification Engine)
   ├─ Stage 1: Detection confidence & ego-vehicle border filtering
   ├─ Stage 2: Temporal persistence & trajectory kinematics
   ├─ Stage 3: Farneback optical flow motion consistency
   ├─ Stage 4: Spatial vehicle interaction & proximity overlap
   └─ Stage 5: Weighted confidence fusion (τ ≥ 0.45, ≥ 2 verified frames)
   │
   ├── [UNVERIFIED] ──► Bypass SASA & SEDE ──► NO ACCIDENT DETECTED
   │
   ▼ [VERIFIED]
SASA (Smart Accident Severity Analyzer)
   └─ Multi-feature visual severity index → LOW / MODERATE / HIGH / CRITICAL
   │
   ▼
SEDE (Smart Emergency Decision Engine)
   └─ Deterministic policy matrix → Operator & emergency dispatch advisory
   │
   ▼
FASTAPI BACKEND + FORENSIC TIMELINE DASHBOARD + JSON / ANNOTATED MP4 OUTPUT
```

---

## 3. Research Architecture & Implementation Transparency

### Primary Research Foundation (Base Paper)
- **Title**: *A Deep Learning Framework for Traffic Accident Detection Based on Improved YOLO11*
- **Authors**: Weijun Li, Liyan Huang, Xiaofeng Lai (*Vehicles*, 2025)
- **Base Paper Concepts**: Multi-head Linear Local Attention (**MLLA**), Asymptotic Feature Pyramid Network (**AFPN**), and **Focaler-IoU** loss.

### Honest Implementation Status (Code vs. Saved Checkpoints)
To maintain strict research reproducibility and transparency:

| Component | File / Artifact | Actual Implementation & Role |
| :--- | :--- | :--- |
| `MLLAAttention` | `backend/detection/yolov11_amf.py` | Standalone PyTorch `nn.Module` reference implementation. Shape/forward pass verified in `train_yolo11_amf.py` (`verify_amf_architecture()`) and instantiated on `YOLO11AMFDetector.__init__`. |
| `AFPNBlock` | `backend/detection/yolov11_amf.py` | Standalone PyTorch `nn.Module` reference implementation. Shape/forward pass verified before training and instantiated on `YOLO11AMFDetector.__init__`. |
| `focaler_iou_loss` | `backend/detection/yolov11_amf.py` | Standalone PyTorch loss function reference implementation verified in `verify_amf_architecture()`. |
| `backend/models/yolov11_amf.pt` | Trained checkpoint (`5.21 MB`) | Trained from Ultralytics `yolo11n.pt` (`yolo11n.yaml`, 320 modules) for **30 epochs** at `imgsz=640`, `batch=16`, `iou=0.45`, `lr0=0.003`, `lrf=0.01`, `mosaic=0.5`. **Does not** embed `MLLAAttention`, `AFPNBlock`, or custom `focaler_iou_loss` inside the Ultralytics model graph/loss loop. |
| `backend/models/yolov11_baseline.pt` | Trained checkpoint (`5.19 MB`) | Trained from Ultralytics `yolo11n.pt` (`yolo11n.yaml`, 320 modules) for **20 epochs** at `imgsz=480`, `batch=8`, `iou=0.70`, `lr0=0.01`, `lrf=0.01`, `mosaic=1.0`. |

### Proposed Research Modules
1. **AMAVE V2 (`backend/amave/`)**:
   - Filters dashcam hood/border artifacts (`is_ego_border_box`).
   - Evaluates temporal continuity (`MIN_PERSISTENCE_FRAMES = 2` within a `10`-frame sliding window).
   - Computes Farneback optical flow inside candidate boxes vs. a surrounding background ring to separate laminar highway traffic from chaotic collision kinetics and post-impact stoppage.
   - Fuses `det` (`0.30`), `temp` (`0.20`), `motion` (`0.30`), and `inter` (`0.20`); requires `fused_score >= 0.45` across at least `MIN_VERIFIED_FRAMES = 2` windows.
2. **SASA (`backend/sasa/`)**:
   - Strictly gated behind `AMAVE` verification (`NOT_EXECUTED` / `BYPASSED` for unverified events).
   - Evaluates collision type, rollover flag, detected box count, optical-flow motion intensity, and roadway layout (falling back transparently to documented defaults when scene metadata is absent).
3. **SEDE (`backend/sede/`)**:
   - Strictly gated behind `AMAVE` + `SASA`.
   - Maps fused verification score and SASA severity onto an auditable policy matrix (`RECORD_EVENT`, `NOTIFY_CONTROL_ROOM`, `DISPATCH_EMERGENCY_SERVICES`, `MAXIMUM_PRIORITY_DISPATCH`) in advisory mode only.

---

## 4. Technologies Used

- **Language & ML**: Python 3.10+, PyTorch 2.0+, Ultralytics YOLO11, OpenCV (`cv2`), NumPy, Pandas, Scikit-learn
- **Backend & API**: FastAPI, Uvicorn, Pydantic, `python-multipart`
- **Frontend**: HTML5, CSS3, Vanilla JavaScript (`frontend/`)
- **Visualization & Reporting**: Matplotlib, Seaborn, Pillow
- **Testing**: `pytest`, `httpx`, `requests`

---

## 5. Repository Structure

```text
├── backend/
│   ├── main.py                  # FastAPI server entry point & static frontend mount
│   ├── config.py                # Paths, device selection, AMAVE/SASA/SEDE thresholds
│   ├── requirements.txt         # Core runtime dependencies
│   ├── api/routes.py            # REST endpoints (/health, /detect, /analyze, /metrics)
│   ├── detection/               # YOLO11 baseline, YOLO11-AMF wrapper, video & temporal buffer
│   ├── amave/                   # AMAVE V2 verification engine & confidence fusion
│   ├── sasa/                    # Smart Accident Severity Analyzer
│   ├── sede/                    # Smart Emergency Decision Engine
│   ├── training/                # Dataset prep, training, evaluation & chart scripts
│   ├── experiments/             # Diagnostic and small-object analysis scripts
│   └── models/                  # Trained .pt checkpoints
├── frontend/                    # Forensic timeline web console (HTML/CSS/JS)
├── dataset/                     # Metadata CSVs, prepared YOLO splits, evaluation clips (see dataset/README.md)
├── experiments/                 # Saved training run outputs and checkpoints
├── results/                     # Metrics JSONs, plots, reports, annotated sample videos
├── tests/                       # Pytest unit tests + 10-video end-to-end harness
├── project_output_screenshots/  # Captured UI & benchmark screenshots
├── requirements-dev.txt         # Runtime + test dependencies (pytest, httpx, requests)
├── run_backend.bat / .sh        # Backend server launchers
├── run_frontend.bat / .sh       # Browser launchers
├── train_gpu.bat / .sh          # Model training launchers
└── evaluate.bat / .sh           # Ablation evaluation launchers
```

---

## 6. Installation Instructions

### Prerequisites
- Python **3.10+**
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/gowtham280920058825/-Adaptive-Multi-Stage-Accident-Detection.git
cd -Adaptive-Multi-Stage-Accident-Detection
```

### Step 2: Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

For running the application (runtime only):
```bash
pip install -r backend/requirements.txt
```

For running the application **and** the full test suite (`pytest`, `httpx`):
```bash
pip install -r requirements-dev.txt
```

### Step 4: Optional CUDA / GPU Setup (NVIDIA GPUs)
Standard `pip install -r backend/requirements.txt` installs CPU-only PyTorch on Windows. To enable CUDA GPU acceleration, install the CUDA-enabled PyTorch wheel matching your NVIDIA driver **after** creating your virtual environment (for example, CUDA 12.1):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Verify GPU detection:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```
If no GPU is available, `backend/config.py` falls back automatically to CPU.

---

## 7. Model Weights & Dataset Setup

### Model Weights (Included)
All required weights are already included in the repository:
- `backend/models/yolov11_amf.pt` (`5.21 MB`) — Primary detector checkpoint
- `backend/models/yolov11_baseline.pt` (`5.19 MB`) — Baseline comparison checkpoint
- `backend/models/yolov11_amf_prev20ep.pt` (`5.19 MB`) — Earlier 20-epoch checkpoint
- `yolo11n.pt`, `yolov8n.pt`, `yolo26n.pt` — Base Ultralytics checkpoints

### Dataset Setup
The complete raw dataset (~19.7 GB) is intentionally excluded from GitHub, while all metadata, prepared YOLO train/val/test splits (`1,680` / `348` / `400` paired images & labels), `10` evaluation clips, `19` representative real videos, and `5` synthetic samples are tracked directly in `dataset/`.

See [dataset/README.md](dataset/README.md) for the exact inventory and instructions to restore the full 19.7 GB dataset locally.

---

## 8. Running the Application (Backend & Frontend)

### Start the Server
- **Windows:** run `run_backend.bat`
- **Linux / macOS:** run `./run_backend.sh`
- **Cross-platform command:**
  ```bash
  python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
  ```

### Open the Dashboard
1. Navigate to `http://127.0.0.1:8000` in your browser (or run `run_frontend.bat` / `./run_frontend.sh`).
2. Click **SELECT VIDEO FILE** and choose a sample video (for example `dataset/evaluation_videos/accident/__WFqm4i3vE_00.mp4` or `dataset/evaluation_videos/non_accident/normal_traffic_clear.mp4`).
3. Click **`INITIALIZE FORENSIC INVESTIGATION →`**.
4. Inspect each stage (`VIDEO INPUT` → `DETECTION` → `AMAVE` → `SASA` → `SEDE` → `FINAL`), replay the annotated MP4 stream, or export the JSON report.

---

## 9. API Documentation & Sample Inference

Interactive Swagger docs are available at `http://127.0.0.1:8000/docs` while the server is running.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | System status, GPU/CPU device info, loaded models & modules |
| `GET` | `/api/model-info` | Active model and hardware metadata |
| `GET` | `/api/metrics` | Evaluation metrics JSON (`results/metrics/ablation_study_results.json`) |
| `GET` | `/api/experiment-summary` | Ablation comparison summary |
| `POST` | `/api/detect` | Single-frame image candidate detection (`multipart/form-data`) |
| `POST` | `/api/analyze` | Full video pipeline (`YOLO11-AMF -> AMAVE -> SASA -> SEDE`) |
| `GET` | `/api/results/{folder}/{filename}` | Retrieve generated annotated videos or JSON reports |

### Sample `curl` Inference
```bash
# Single image detection
curl -X POST "http://127.0.0.1:8000/api/detect" ^
  -F "file=@dataset/prepared/images/test/0isplBZQftg_00_f120_acc.jpg"

# Full end-to-end video pipeline
curl -X POST "http://127.0.0.1:8000/api/analyze" ^
  -F "file=@dataset/evaluation_videos/accident/__WFqm4i3vE_00.mp4"
```
Generated annotated videos are saved under `results/annotated_videos/` and full JSON reports under `results/reports/`.

---

## 10. Testing Commands

### Unit & API Tests (`pytest`)
```bash
pytest tests/test_amave.py tests/test_backend.py tests/test_sasa.py tests/test_sede.py -v
```

### End-to-End 10-Video Detection Harness
```bash
python tests/test_end_to_end_detection.py
```
Outputs the full confusion matrix and per-video summary to `results/metrics/test_harness_evaluation.json`.

### Training & Evaluation Scripts
See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for running `prepare_dataset.py`, `train_yolo11.py`, `train_yolo11_amf.py`, and `evaluate.py`.

---

## 11. Known Limitations

- **No Ground-Truth Severity or Dispatch Labels**: Public crash metadata does not contain clinical injury severity or real emergency dispatch labels, so `SASA` and `SEDE` are evaluated via deterministic policy and feature-contribution tests rather than supervised classification accuracy.
- **Impact-Frame Bounding Box Alignment**: Dataset spatial bounding boxes are annotated around accident impact frames.
- **Custom AMF Modules vs. Ultralytics Checkpoint**: As documented in Section 3, `MLLAAttention`, `AFPNBlock`, and `focaler_iou_loss` exist as standalone verified PyTorch modules, whereas the saved `yolov11_amf.pt` checkpoint uses the standard Ultralytics `yolo11n.yaml` graph trained with customized hyperparameters.
- **Simulated Emergency Alerts**: `SEDE` outputs advisory recommendations for control-room operators and never triggers real emergency calls.
- **Recall on Challenging Low-Light / Distant Crashes**: On the 10-video end-to-end harness, precision is `100.00%` (`0` false positives) and recall is `71.43%` (`5/7` accident videos verified; `2` low-confidence highway/night cases missed).
