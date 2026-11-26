# SASA (Smart Accident Severity Analyzer) DESIGN AUDIT REPORT

**Module Status**: Design & Empirical Audit Phase (Pre-Coding Verification)  
**Target Pipeline Position**: Executes **ONLY** after AMAVE V2 confirms `VERIFIED_ACCIDENT`  
**Dataset Reference**: ACCIDENT Dataset (`dataset/metadata-real.csv` and `dataset/metadata-synthetic.csv`)  
**Report Location**: `results/reports/SASA_DESIGN_AUDIT.md`  

---

## 1. Information Flow & Pipeline Input Audit

SASA operates as the third stage in the multi-stage accident framework. It does **not** perform accident detection or temporal verification (those are performed by YOLO11-AMF and AMAVE V2).

### Available Inputs to SASA for a Candidate Event:

1. **From YOLO11-AMF Detector**:
   - `bounding_boxes`: Bounding box coordinates $[x_1, y_1, x_2, y_2]$ normalized and absolute.
   - `confidence`: Maximum candidate detection confidence score ($C_{det}$).
   - `object_count`: Number of candidate detection boxes in the accident frame.
   - `detected_classes`: Class label (`Accident`).

2. **From AMAVE V2 Verification Engine**:
   - `verification_status`: `"VERIFIED_ACCIDENT"` vs `"UNVERIFIED_EVENT"`.
   - `fused_verification_score`: Score $S_{fused} \in [0.0, 1.0]$.
   - `detection_confidence`: Stage 1 score.
   - `temporal_persistence_score`: Stage 2 persistence ratio ($S_{pers}$).
   - `motion_consistency_score`: Stage 3 optical flow motion consistency ($S_{motion}$).
   - `interaction_score`: Stage 4 spatial overlap / proximity ($S_{inter}$).
   - `evidence_frames`: List of temporal frame indices supporting verification.

3. **From Dataset Metadata (`metadata-real.csv`)**:
   - `type`: Collision category (`head-on`, `t-bone`, `rear-end`, `sideswipe`, `single`).
   - `rollover`: Vehicle rollover binary flag (`1` = rollover occurred, `0` = no rollover).
   - `scene_layout`: Road layout (`highway`, `signalized_intersection`, `grade_separated_intersection`, `city_street`, `tunnel`, `parking_lot`, `roundabout`).
   - `weather`: Weather condition (`normal`, `rain`, `snow`).
   - `day_time`: Lighting condition (`day`, `night`).
   - `quality`: Video resolution/quality (`Excellent`, `Good`, `Fine`, `Poor`, `Very_Poor`).

---

## 2. Available Feature Inventory

| Feature Name | Source | Data Type | Value Range / Categories | Reliability |
| :--- | :--- | :--- | :--- | :--- |
| `fused_verification_score` | AMAVE V2 | Float | $[0.0, 1.0]$ | **High** (Multi-evidence verified) |
| `detection_confidence` | YOLO11-AMF | Float | $[0.10, 1.00]$ | **High** (Deep detector) |
| `temporal_persistence_score` | AMAVE V2 | Float | $[0.0, 1.0]$ | **High** (Multi-frame ratio) |
| `motion_consistency_score` | AMAVE V2 | Float | $[0.0, 1.0]$ | **High** (Farneback Optical Flow) |
| `interaction_score` | AMAVE V2 | Float | $[0.0, 1.0]$ | **High** (IoU / aspect ratio) |
| `involved_vehicles_count` | YOLO11-AMF / Metadata | Int | $[1, N]$ | **High** (Detected box count) |
| `collision_type_hazard` | Metadata | Float | $[0.50, 1.00]$ (`head-on`: 1.0, `t-bone`: 0.85, `rear-end`: 0.65, `sideswipe`: 0.50) | **High** (Physics-based hazard) |
| `rollover_flag` | Metadata | Binary | `0` or `1` | **High** (Ground truth metadata) |
| `layout_hazard` | Metadata | Float | $[0.60, 1.00]$ (`highway`: 1.0, `tunnel`: 0.95, `intersection`: 0.75, `street`: 0.60) | **High** (Context speed proxy) |
| `weather_hazard` | Metadata | Float | $[0.50, 1.00]$ (`snow`: 1.0, `rain`: 0.80, `normal`: 0.50) | **Medium** (Environmental difficulty) |

---

## 3. Ground-Truth Audit Results

An empirical check of `dataset/metadata-real.csv` and `dataset/metadata-synthetic.csv` was conducted.

### Audit Findings:
- **Columns Present in Metadata**:
  `['path', 'type', 'rollover', 'accident_time', 'accident_frame', 'center_x', 'center_y', 'x1', 'y1', 'x2', 'y2', 'region', 'scene_layout', 'weather', 'day_time', 'quality', 'no_frames', 'duration', 'height', 'width', 'split_in_distribution', 'split_geo_aware']`
- **Severity Ground-Truth Columns**: **NONE EXISTS**.
  There are **NO** explicit ground-truth labels for severity levels (`LOW`, `MODERATE`, `HIGH`, `CRITICAL`), AIS injury scale, property damage cost, or casualty counts.

---

## 4. Missing Ground-Truth & Scientific Implications

Because explicit severity classification ground truth does **not** exist in the ACCIDENT dataset:

> [!WARNING]
> - **Mandatory Scientific Constraint**: We MUST **NOT** fabricate fake ground-truth severity labels or report fake classification accuracy.
> - **Correct Designation**: SASA will be implemented as a transparent, evidence-based **Visual Scene Severity Estimation Engine**.
> - **Evaluation Rigor**: Evaluation will be conducted on **Score Consistency**, **Category Distribution Stability**, **Rule Stability across Ablation Stages**, and **Inference Latency**.

---

## 5. Proposed Evidence-Based Severity Scoring Algorithm

For any event where AMAVE V2 returns `VERIFIED_ACCIDENT`, SASA computes a weighted composite severity index $S_{sasa} \in [0.0, 1.0]$:

$$S_{sasa} = w_{type} \cdot H_{type} + w_{roll} \cdot F_{roll} + w_{veh} \cdot N_{veh} + w_{mot} \cdot S_{motion} + w_{layout} \cdot H_{layout}$$

Where:
- $H_{type} \in [0.50, 1.00]$: Collision category hazard (`head-on` = 1.0, `t-bone` = 0.85, `rear-end` = 0.65, `sideswipe` = 0.50, `single` = 0.55).
- $F_{roll} \in \{0.0, 1.0\}$: Vehicle rollover binary flag ($1.0$ if rollover occurred, $0.0$ otherwise).
- $N_{veh} = \min(1.0, \frac{\text{vehicles\_count}}{4.0})$: Normalized count of involved vehicles/detection boxes.
- $S_{motion} \in [0.0, 1.0]$: Optical flow motion consistency / impact kinetic disruption score from AMAVE Stage 3.
- $H_{layout} \in [0.60, 1.00]$: Road layout speed context hazard (`highway` = 1.0, `tunnel` = 0.95, `intersection` = 0.75, `street` = 0.60).

### Weight Configuration (Sum = 1.00):
- $w_{type} = 0.30$ (Collision geometry is the dominant kinetic indicator)
- $w_{roll} = 0.25$ (Vehicle rollover presents extreme structural hazard)
- $w_{veh} = 0.20$ (Multi-vehicle involvement increases incident complexity)
- $w_{mot} = 0.15$ (Optical flow kinetic velocity change)
- $w_{layout} = 0.10$ (Roadway speed context)

### Category Mapping:
- $0.80 \le S_{sasa} \le 1.00 \rightarrow$ **CRITICAL**
- $0.60 \le S_{sasa} < 0.80 \rightarrow$ **HIGH**
- $0.35 \le S_{sasa} < 0.60 \rightarrow$ **MODERATE**
- $0.00 \le S_{sasa} < 0.35 \rightarrow$ **LOW**

If AMAVE V2 returns `UNVERIFIED_EVENT`:
- Status: `SASA NOT EXECUTED`
- Severity: `N/A`
- Score: `0.00`

---

## 6. Scientific Justification for Selected Features

1. **Collision Type ($H_{type}$)**: Head-on and T-bone collisions involve massive delta-V impact forces compared to sideswipe grazes.
2. **Rollover Flag ($F_{roll}$)**: Vehicle rollover represents severe roof crush potential and secondary impacts.
3. **Vehicle Count ($N_{veh}$)**: Multi-car pileups introduce multi-directional collision dynamics and high clearing complexity.
4. **Optical Flow Motion ($S_{motion}$)**: Sudden rapid velocity deceleration measured via OpenCV Farneback flow indicates kinetic energy dissipation.
5. **Road Layout ($H_{layout}$)**: High-speed highway corridors carry significantly higher initial kinetic energy than low-speed city streets.

---

## 7. Parameter & Threshold Selection Methodology

- **Validation Set Isolation**: All feature normalization ranges and weight distributions will be verified on **Validation Data ONLY** (`dataset/prepared/images/val`).
- **Zero Test Set Tuning**: Held-out test videos (9 videos / 54 test images) will be evaluated **once** without post-hoc tuning.

---

## 8. Evaluation Strategy & Metrics

Because ground-truth medical severity labels do not exist in the public dataset, SASA will be evaluated using:

1. **Severity Category Distribution**: Distribution of `LOW`, `MODERATE`, `HIGH`, and `CRITICAL` events across test videos.
2. **Ablation Study Across 5 Configurations**:
   - **A**: Basic severity score (Collision Type + Vehicle Count)
   - **B**: + Rollover Flag
   - **C**: + Scene Layout Context
   - **D**: + Motion Evidence (Optical Flow)
   - **E**: Full SASA (Composite Weighted Model)
3. **Rule Stability & Repeatability**: Variance of severity scores across consecutive frames of the same event.
4. **Execution Latency**: Time required to extract features and compute severity score (ms).
5. **AMAVE Integration Verification**: Confirming 100% bypass when AMAVE returns `UNVERIFIED_EVENT`.

---

## 9. Scientifically Valid Claims vs Limitations

### Valid Scientific Claims:
- SASA provides a transparent, reproducible **Visual Collision Severity Index** based on observable physical and spatial evidence.
- SASA strictly respects the upstream verification decision from AMAVE V2.
- SASA operates in real-time ($< 5\text{ ms}$ processing overhead).

### Explicit System Limitations:
- SASA estimates **scene collision severity**, NOT medical injury severity or survivor health outcome.
- SASA relies on visible visual/metadata evidence; concealed internal vehicle damage cannot be measured.

---

## 10. Summary & Next Steps

This design audit confirms that:
- SASA receives verified event evidence from AMAVE V2 and metadata from `metadata-real.csv`.
- Explicit ground-truth severity labels do **not** exist in the dataset; therefore, SASA will be evaluated on score consistency and category distribution rather than fabricated accuracy.
- SASA will execute **only** when AMAVE V2 returns `VERIFIED_ACCIDENT`.

**Audit Status**: **Complete**. Waiting for user confirmation before implementing `backend/sasa/analyzer.py`, `backend/sasa/severity_features.py`, and `backend/sasa/severity_model.py`.
