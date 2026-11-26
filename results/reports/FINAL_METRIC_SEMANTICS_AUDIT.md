# FINAL METRIC SEMANTICS AUDIT REPORT

**Project Title**: Adaptive Multi-Stage Framework for Vision-Based Road Accident Detection and Intelligent Emergency Alert System  
**Audit Purpose**: Academic Rigor & Metric Comparability Disambiguation  
**Evaluation Dataset**: ACCIDENT Dataset (60-Video Split: 42 Train / 9 Val / 9 Held-Out Test Videos; 54 Test Images)  
**Report Location**: `results/reports/FINAL_METRIC_SEMANTICS_AUDIT.md`  

---

## Executive Summary & Disambiguation Statement

To ensure complete academic transparency and scientific rigor for paper publication and research presentations, this audit establishes a strict separation between four distinct metric domains in the proposed framework:

> [!IMPORTANT]
> 1. **Detector Metrics** (YOLO11n, YOLO11-AMF) evaluate **bounding-box object detection proposal quality** via spatial IoU ($\ge 0.50$).
> 2. **Verification Metrics** (AMAVE V1, AMAVE V2) evaluate **post-detection temporal event verification** (false alarm filtering over consecutive frames).
> 3. **Severity Estimation Metrics** (SASA) evaluate **visual scene impact scoring** ($S_{sasa} \in [0.0, 1.0]$). Because dataset metadata lacks explicit medical/damage labels, **no fake classification accuracy is reported**.
> 4. **Decision Support Metrics** (SEDE) evaluate **emergency priority response policy mapping**. Evaluated via deterministic **Policy Validation Tests (Cases 1–5)**, NOT fake decision accuracy.

---

## Section A: Object Detection Performance

- **Evaluation Unit**: Bounding Box Candidate Proposal / Image Spatial IoU ($\text{IoU} \ge 0.50$).
- **Ground Truth Source**: Ultralytics YOLO ground-truth bounding box labels (`dataset/prepared/labels/test/`).
- **Test Sample Size**: 54 Held-Out Test Images across 9 Test Videos.
- **Metric Formulas**:
  - $\text{Precision} = \frac{TP_{box}}{TP_{box} + FP_{box}}$
  - $\text{Recall} = \frac{TP_{box}}{TP_{box} + FN_{box}}$
  - $\text{mAP50} = \text{Mean Average Precision at IoU } 0.50$
  - $\text{mAP50-95} = \text{Mean Average Precision integrated across IoU } [0.50, 0.95]$
  - $\text{F1-Score} = \frac{2 \cdot P \cdot R}{P + R}$

### Empirical Detection Results:

| Model Architecture | Precision | Recall | mAP50 | mAP50-95 | F1-Score | Inference Latency / FPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLO11n Baseline** | 0.3123 | 0.0741 | 0.0626 | 0.0184 | 0.1197 | 8.47 ms (118.06 FPS) |
| **YOLO11-AMF Base Paper** | **0.3730** | **0.0741** | **0.0631** | **0.0186** | **0.1236** | **8.47 ms (118.06 FPS)** |

*Scientific Note: YOLO11-AMF incorporates MLLA, AFPN, and Focaler-IoU, improving detection Precision (+0.0607), mAP50 (+0.0005), mAP50-95 (+0.0002), and F1-score (+0.0039).*

---

## Section B: Accident Verification Performance (AMAVE)

- **Evaluation Unit**: Frame-level Event Verification Decision (`VERIFIED_ACCIDENT` vs `UNVERIFIED_EVENT`).
- **Ground Truth Source**: Binary Frame Annotation (27 Accident Frames / 27 Normal Frames).
- **Test Sample Size**: 54 Held-Out Test Frames across 9 Test Videos.
- **Metric Formulas**:
  - $\text{Verification Precision} = \frac{TP_{event}}{TP_{event} + FP_{event}}$
  - $\text{Verification Recall} = \frac{TP_{event}}{TP_{event} + FN_{event}}$
  - $\text{Verification F1-Score} = \frac{2 \cdot P \cdot R}{P + R}$
  - $\text{Verification Accuracy} = \frac{TP_{event} + TN_{event}}{N_{total}}$
  - $\text{False-Positive Rate (FPR)} = \frac{FP_{event}}{FP_{event} + TN_{event}}$
  - $\text{False-Alert Reduction} = \frac{FP_{raw} - FP_{amave}}{FP_{raw}} \times 100\%$

### Empirical Verification Results & Ablation (AMAVE V2):

| Module Configuration | Verification Precision | Verification Recall | Verification F1 | Verification Accuracy | FPR | False-Alert Reduction | Verification Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Config A: YOLO11-AMF Only** | 0.7500 | 0.2222 | 0.3429 | 0.5741 | 0.0741 ($2/27$) | Baseline | 38.97 ms |
| **Config B: + Temporal Persistence** | 0.7500 | 0.2222 | 0.3429 | 0.5741 | 0.0741 ($2/27$) | Baseline | 38.97 ms |
| **Config C: + Motion Consistency (Optical Flow)** | **0.8000** | 0.1481 | 0.2500 | 0.5556 | **0.0370** ($1/27$) | **50.0% Reduction** | 38.97 ms |
| **Config D: + Collision Interaction** | **0.8000** | 0.1481 | 0.2500 | 0.5556 | **0.0370** ($1/27$) | **50.0% Reduction** | 38.97 ms |
| **Config E: Full Refined AMAVE V2 Engine** | **0.7143** | **0.1852** | **0.2941** | **0.5556** | 0.0741 ($2/27$) | 0.0% (Full Engine) | **38.97 ms** |
| **Original AMAVE_V1 (Baseline Threshold 0.50)** | 0.6000 | 0.1111 | 0.1875 | 0.5185 | 0.0741 ($2/27$) | 0.0% | 37.54 ms |

### FPR Audit Clarification:
- **FPR = 0.0741**: Corresponds to 2 false positives out of 27 normal test frames in Configs A, B, E, and AMAVE_V1.
- **FPR = 0.0370**: Corresponds to **1 false positive out of 27 normal test frames** in **Configs C and D** (where Optical Flow Motion Consistency and Collision Interaction filters suppressed 1 additional false alarm, achieving a **50.0% reduction in false alerts**).

---

## Section C: Visual Scene Severity Estimation (SASA)

- **Evaluation Unit**: Verified Accident Event (Executed ONLY when AMAVE V2 returns `VERIFIED_ACCIDENT`).
- **Ground Truth Status**: **No explicit ground-truth medical or property damage severity labels exist in the public dataset metadata.**
- **Comparability Statement**: SASA metrics measure **visual impact evidence distribution and score stability**; they are **NOT comparable** to detection or verification classification accuracy.

### Empirical Severity Estimation Results:

- **Formula**: $S_{sasa} = 0.30 \cdot H_{type} + 0.25 \cdot F_{roll} + 0.20 \cdot N_{veh} + 0.15 \cdot S_{motion} + 0.10 \cdot H_{layout}$
- **Score Range**: [$0.3725$, $0.5832$] (Mean: **0.4993**, Std Dev: **0.0727**)
- **Executed Events Count**: 7 Verified Frames
- **Bypassed Events Count**: 47 Unverified Frames (`sasa_status`: `NOT_EXECUTED`, `severity`: `N/A`)
- **Severity Category Distribution**: `LOW`: 0, `MODERATE`: 7, `HIGH`: 0, `CRITICAL`: 0
- **Average Processing Overhead**: **0.38 ms** per event

---

## Section D: Emergency Decision Support (SEDE)

- **Evaluation Unit**: Verified Accident Event (Executed ONLY when AMAVE V2 returns `VERIFIED_ACCIDENT` and SASA is `EXECUTED`).
- **Ground Truth Status**: **No ground-truth emergency dispatch decision labels exist in public datasets.**
- **Comparability Statement**: SEDE metrics reflect **policy recommendation mapping**; they are **NOT comparable** to model classification accuracy.

### Deterministic Policy Validation Suite Results:

| Scenario Test ID | Description | Expected Status / Priority | Evaluated Status / Priority | Policy Test Result |
| :--- | :--- | :--- | :--- | :--- |
| **CASE 1** | Unverified Candidate Event | `NOT_EXECUTED` / `N/A` | `NOT_EXECUTED` / `N/A` | **PASSED** |
| **CASE 2** | Verified Minor Collision | `EXECUTED` / `LOW` | `EXECUTED` / `LOW` | **PASSED** |
| **CASE 3** | Verified Moderate Intersection | `EXECUTED` / `MODERATE` | `EXECUTED` / `MODERATE` | **PASSED** |
| **CASE 4** | Verified High-Impact Collision | `EXECUTED` / `HIGH` | `EXECUTED` / `HIGH` | **PASSED** |
| **CASE 5** | Verified Critical Rollover | `EXECUTED` / `CRITICAL` | `EXECUTED` / `CRITICAL` | **PASSED** |

### Test Set Priority Distribution & Overhead:
- **Emergency Priority Distribution**: `LOW`: 0, `MODERATE`: 6, `HIGH`: 1, `CRITICAL`: 0
- **Average Processing Overhead**: **0.05 ms** per event

---

## Master Metric Comparability Matrix

| Framework Stage | Evaluation Domain | Target Unit | Ground Truth Source | Comparable to Object Detection? |
| :--- | :--- | :--- | :--- | :--- |
| **YOLO11n / YOLO11-AMF** | Object Detection | Bounding Box / Image | YOLO GT Bounding Boxes | **YES** (Standard Object Detection) |
| **AMAVE V1 / AMAVE V2** | Event Verification | Temporal Frame Sequence | Binary Frame Ground Truth | **NO** (Post-detection verification) |
| **SASA Engine** | Severity Estimation | Verified Event | None (Evidence Index) | **NO** (Estimated Visual Index) |
| **SEDE Engine** | Decision Support | Verified Event | None (Policy Scenarios) | **NO** (Rule-based Decision Support) |
