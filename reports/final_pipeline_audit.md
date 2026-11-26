# Final Comprehensive Pipeline Audit & Scientific Verification Report
## Adaptive Multi-Stage Framework for Vision-Based Road Accident Detection

---

### Executive Summary

A critical end-to-end audit of the entire road accident detection pipeline was performed without fabricating metrics or forcing positive predictions. The frozen deployed detector (`backend/models/yolov11_amf.pt`), temporal verification engine (AMAVE V2), severity analyzer (SASA), and decision engine (SEDE) were audited across both daytime and night accident scenes.

**Key Findings:**
1. **Root Detector Limitation Identified:** The frozen YOLO11-AMF model exhibits high precision on prominent, high-speed multi-vehicle collisions (e.g. `__WFqm4i3vE_00.mp4`, `0NixrF6Mt00_0_00.mp4`, `-6SQSDj8cYU_00.mp4`), but has limited feature sensitivity on **distant, small-scale ($29 \times 18$ px) or dark-shoulder single-vehicle accidents** (e.g. `_9tp71S2Gwg_00.mp4`, `_0RDZgCKPGo_00.mp4`), where raw YOLO confidence peaks at only **$2.6\%$** and **$10.2\%$** respectively.
2. **AMAVE False-Positive Suppression Working as Designed:** AMAVE correctly rejects foreground highway vehicles (such as passing green trucks and normal SUVs) that trigger spurious candidate detections ($20\% - 47\%$) under laminar highway motion ($\text{coherence} > 0.90$). It correctly maintained a **$0.0\%$ False Positive Rate** across all tested normal traffic videos.
3. **Misleading 100% Metric Removed:** The previously displayed "100% Event Accuracy" was derived from a small, curated 10-video sample. It has been replaced with the honest, empirical **10-Video Test Harness Benchmark ($80.0\%$ Accuracy: $\text{TP}=5, \text{TN}=3, \text{FP}=0, \text{FN}=2$)**. Frame-level YOLO test metrics ($\text{Precision}=40.94\%, \text{Recall}=17.33\%, \text{mAP50}=11.48\%$) are now strictly separated.
4. **UI Reporting Bugs Resolved:** Proposal counts are decoupled from AMAVE verification status, and the static file routing conflict causing HTTP 404 errors on annotated video streams was resolved.

---

### 1. Root Cause Analysis

The inconsistent detection performance across accident videos stems from two primary technical causes:

1. **Detector Sensitivity on Small & Distant Targets (Object Scale vs. Resolution):**
   - In highway surveillance footage (e.g., `_9tp71S2Gwg_00.mp4`, $960 \times 720$), distant collisions occupy small bounding boxes ($29 \times 18$ pixels, $< 0.1\%$ of the frame area).
   - When resized to $640 \times 640$ for YOLO11 inference, high-frequency spatial gradients describing deformed vehicle panels are lost.
   - The detector outputs only **$2.6\%$ raw confidence** on the actual accident at Frame 83, failing to generate a candidate proposal ($\tau \ge 0.20$).
   - Concurrently, normal moving vehicles in the foreground trigger candidate proposals ($20\% - 46.9\%$).
2. **Strict Verification Gating on Non-Interacting Candidates:**
   - Because the true accident vehicle was never proposed by YOLO, AMAVE V2 receives only the foreground passing vehicles.
   - AMAVE evaluates their Farneback optical flow and spatial relationships: finding unidirectional laminar motion ($\text{coherence} = 0.95$) and no collision contact ($S_{inter} = 0.15$).
   - AMAVE correctly rejects these non-accident candidates ($S_{fused} = 0.4442 < 0.4500$).
   - The video is thus classified as `NO ACCIDENT DETECTED` (False Negative).

---

### 2. Exact Failing Component

- **Primary Failing Component:** **Raw YOLO11-AMF Detector (`backend/models/yolov11_amf.pt`)**
  - Fails to generate candidate bounding box proposals above the $0.20$ threshold for small, distant, or unlit stationary single-vehicle hazards ($2.6\%$ confidence on `_9tp71S2Gwg_00.mp4`, $10.2\%$ on `_0RDZgCKPGo_00.mp4`).
- **Secondary Failing Component (Now Fixed):** **Frontend Reporting Logic (`frontend/script.js`)**
  - Line 298 previously contained `v2CandidateCount.textContent = data.accident_detected ? candidateCount : 0;`, zeroing proposal counts on any AMAVE-unverified video, displaying "Proposals: 0" alongside "Detection Confidence: 31.4% / 40.0%".

---

### 3. Empirical Evidence & Diagnostic Data

#### A. Raw YOLO Inference on Daytime Accident Video (`_9tp71S2Gwg_00.mp4`)
- **Total Frames Analyzed:** 440 frames (every frame tested at $\tau = 0.01$)
- **Peak YOLO Confidence:** $0.4690$ ($46.90\%$) at Frame 387 (triggered by a green truck driving normally in the left lane)
- **Actual Ground Truth Accident Location:** Frame 83, coordinates $[418, 220, 448, 239]$
- **Raw YOLO Confidence on Actual Accident:** **$0.0257$ ($2.57\%$)**

#### B. Threshold Sweep on Daytime Video (`_9tp71S2Gwg_00.mp4`)

| Threshold ($\tau$) | Detection Frames | Total Bounding Boxes | Max Confidence | Detection Source |
| :---: | :---: | :---: | :---: | :--- |
| **0.01** | 440 | 1090 | 0.4690 | Background / Road / All vehicles |
| **0.05** | 124 | 135 | 0.4690 | Moving highway vehicles |
| **0.10** | 67 | 67 | 0.4690 | Moving highway vehicles |
| **0.15** | 52 | 52 | 0.4690 | Moving highway vehicles |
| **0.20** | **38** | **38** | **0.4690** | Moving highway vehicles (GT accident missed: 2.6%) |
| **0.30** | 17 | 17 | 0.4690 | Foreground trucks & SUVs |
| **0.40** | 3 | 3 | 0.4690 | Green commercial truck |
| **0.50** | 0 | 0 | 0.0000 | None |

#### C. Multi-Scale Inference Test on Actual Accident Frame (Frame 83)

| Resolution (`imgsz`) | Box on Actual Accident Vehicle | Confidence on Actual Accident | Foreground Structure False Confidence |
| :---: | :---: | :---: | :---: |
| **640** | $[418.6, 220.8, 448.5, 240.6]$ | **2.57%** | 1.95% |
| **960** | $[420.5, 221.2, 449.8, 240.5]$ | **2.90%** | 27.21% (Overhead Gantry) |
| **1280** | $[426.4, 227.9, 471.7, 261.6]$ | **2.49%** | 48.94% (Overhead Gantry) |

**Conclusion:** Scaling image resolution alone cannot resolve the sensitivity gap for distant daytime highway accidents with the current model weights, as the detector lacks specific training on sub-$32$px accident vehicles.

---

### 4. Applied Pipeline Corrections

1. **Frozen Component Manifest (`reports/system_freeze_manifest.json`):**
   - Documented SHA-256 hashes and file sizes for `yolov11_amf.pt`, `amave/`, `sasa/`, `sede/`. Model weights remained completely untouched.
2. **Frontend Proposal Decoupling (`frontend/script.js`):**
   - Stage 2 proposal counter now reflects actual candidate proposals returned by the detector, independent of AMAVE's final verdict.
3. **Dedicated UI Metric Separation (`frontend/index.html`):**
   - Relabeled `DETECTION CONFIDENCE` to `RAW MAX YOLO CONFIDENCE`.
   - Added explicit `CANDIDATE PROPOSALS` count in the conclusion intelligence block.
   - Separated Frame-Level YOLO test metrics from Video-Level Benchmark metrics.
4. **Static Route Mount Ordering (`backend/main.py`):**
   - Mounted `/results` before the catch-all frontend static route, eliminating HTTP 404 errors on video stream replay.
5. **Automated Test Harness (`tests/test_end_to_end_detection.py`):**
   - Built a reproducible benchmark suite executing all 10 test videos through the full pipeline without hardcoded or fabricated results.

---

### 5. Automated 10-Video Test Harness Results

The test harness was executed on 10 verified real-world test cases:

| Video Name | Scenario Category | Ground Truth | YOLO Detection | Max Conf | AMAVE Score | AMAVE Status | Final Prediction | Evaluation Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `_9tp71S2Gwg_00.mp4` | Clear Morning Highway | ACCIDENT | 19 frames | 40.0% | 0.4442 | UNVERIFIED | NORMAL | **FALSE NEGATIVE** |
| `__WFqm4i3vE_00.mp4` | High-Speed Collision | ACCIDENT | 56 frames | 57.6% | 0.6339 | VERIFIED | ACCIDENT | **TRUE POSITIVE** |
| `_0RDZgCKPGo_00.mp4` | Night CCTV Highway | ACCIDENT | 16 frames | 31.4% | 0.3743 | UNVERIFIED | NORMAL | **FALSE NEGATIVE** |
| `0NixrF6Mt00_0_00.mp4` | Urban Intersection Crash | ACCIDENT | 15 frames | 84.4% | 0.6428 | VERIFIED | ACCIDENT | **TRUE POSITIVE** |
| `-6SQSDj8cYU_00.mp4` | Signalized T-Bone | ACCIDENT | 154 frames | 72.3% | 0.7706 | VERIFIED | ACCIDENT | **TRUE POSITIVE** |
| `Z4kg2Ev3vhk_00.mp4` | Night Rear-End | ACCIDENT | 44 frames | 70.5% | 0.5838 | VERIFIED | ACCIDENT | **TRUE POSITIVE** |
| `nAXTthLfgtI_00.mp4` | Low-Light Multi-Vehicle | ACCIDENT | 33 frames | 63.4% | 0.5422 | VERIFIED | ACCIDENT | **TRUE POSITIVE** |
| `normal_traffic_clear.mp4` | Normal Daytime Flow | NORMAL | 30 frames | 47.4% | 0.2497 | UNVERIFIED | NORMAL | **TRUE NEGATIVE** |
| `normal_traffic_urban.mp4` | Normal Urban Traffic | NORMAL | 4 frames | 42.1% | 0.3433 | UNVERIFIED | NORMAL | **TRUE NEGATIVE** |
| `test_normal_traffic.mp4` | Normal Highway CCTV | NORMAL | 18 frames | 57.4% | 0.5098 | UNVERIFIED* | NORMAL | **TRUE NEGATIVE** |

*\*Note: `test_normal_traffic.mp4` produced a single transient verified frame, but failed the minimum threshold of $\ge 2$ consecutive verified frames (`MIN_VERIFIED_FRAMES`), correctly resulting in a final decision of NORMAL.*

---

### 6. Official Metrics Breakdown

#### A. Frame-Level YOLO Metrics (Evaluated on Held-Out Test Set: 400 Images)
- **Precision:** $40.94\%$ ($0.4094$)
- **Recall:** $17.33\%$ ($0.1733$)
- **mAP@50:** $11.48\%$ ($0.1148$)
- **mAP@50–95:** $4.56\%$ ($0.0456$)
- **Detection F1-Score:** $0.2436$
- **Frame False Positive Rate (FPR):** $3.70\%$

#### B. Event-Level System Metrics (10-Video Test Harness Benchmark)
- **Total Test Videos:** 10 (7 Accident, 3 Normal)
- **True Positives (TP):** 5
- **True Negatives (TN):** 3
- **False Positives (FP):** 0
- **False Negatives (FN):** 2
- **Event-Level Accuracy:** **$80.00\%$** ($8/10$)
- **Event-Level Precision:** **$100.00\%$** ($5/5$)
- **Event-Level Recall:** **$71.43\%$** ($5/7$)
- **Event-Level F1-Score:** **$83.33\%$** ($0.8333$)
- **Event False Positive Rate (FPR):** **$0.00\%$** ($0/3$)

#### C. Confusion Matrix (Video-Level)

```
                    Predicted ACCIDENT    Predicted NORMAL
Actual ACCIDENT             5 (TP)                2 (FN)
Actual NORMAL               0 (FP)                3 (TN)
```

---

### 7. Known Scientific Limitations & Recommendations

1. **Distant Small-Scale Accidents ($< 32 \times 32$ px):**
   - The current frozen YOLO11-AMF detector cannot reliably generate candidate proposals for accidents occurring far from the camera in wide-angle CCTV views.
   - *Recommendation for future training:* Multi-scale cropping (tiling / SAHI) or specialized training on small accident instances.
2. **Single-Vehicle Off-Road / Shoulder Incidents:**
   - Single-vehicle accidents with stationary post-impact vehicles under poor lighting produce low motion disruption and no multi-vehicle interaction overlap.
   - AMAVE correctly relies on multi-vehicle overlap or confirmed crash deceleration. Standalone stationary vehicles without pre-impact trajectory data are inherently difficult to distinguish from normal parked vehicles.
3. **Detector Pre-Filtering is Essential:**
   - AMAVE V2 cannot verify an accident that YOLO never proposes. Retaining high detector precision without sacrificing recall on small objects requires targeted dataset enrichment rather than arbitrary threshold lowering.
