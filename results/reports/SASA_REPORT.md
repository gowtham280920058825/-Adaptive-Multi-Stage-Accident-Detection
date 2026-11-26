# SASA (Smart Accident Severity Analyzer) EVALUATION REPORT

**Module Status**: **Completed**  
**Scientific Definition**: Visual Scene Severity Estimation Engine  
**Pipeline Integration**: Executes **ONLY** on AMAVE V2 `VERIFIED_ACCIDENT` events  
**Held-Out Test Set**: 54 Test Frames across 9 Test Videos  
**Metrics Output**: `results/metrics/sasa_metrics.json`  
**Visual Evidence Output**: `results/reports/sasa_visualization/`  

---

## 1. Executive Summary & Execution Counts

- **Total Test Frames Evaluated**: 54
- **Verified Accident Events (SASA Executed)**: **7**
- **Unverified Events (SASA Bypassed)**: **47** (sasa_status: `NOT_EXECUTED`)
- **Average SASA Processing Overhead**: **0.38 ms** per event

---

## 2. Test Set Visual Scene Severity Distribution

| Severity Category | Event Count | Percentage |
| :--- | :--- | :--- |
| **LOW** ($S < 0.35$) | 0 | 0.0% |
| **MODERATE** ($0.35 \le S < 0.60$) | 7 | 100.0% |
| **HIGH** ($0.60 \le S < 0.80$) | 0 | 0.0% |
| **CRITICAL** ($S \ge 0.80$) | 0 | 0.0% |

- **Mean Severity Score**: **0.4993**
- **Standard Deviation**: **0.0727**
- **Score Range**: [0.3725, 0.5832]

---

## 3. Approved Scoring Model & Weight Rationale

$$S_{sasa} = 0.30 \cdot H_{type} + 0.25 \cdot F_{roll} + 0.20 \cdot N_{veh} + 0.15 \cdot S_{motion} + 0.10 \cdot H_{layout}$$

1. **Collision Type Hazard ($H_{type}$, 30%)**: Physics-based hazard weighting (`head-on` = 1.0, `t-bone` = 0.85, `rear-end` = 0.65, `sideswipe` = 0.50).
2. **Rollover Flag ($F_{roll}$, 25%)**: Vehicle rollover flag ($1.0$ if rollover occurred, $0.0$ otherwise).
3. **Vehicle Involvement ($N_{veh}$, 20%)**: Normalized detected vehicle count $\min(1.0, \text{vehicles}/4.0)$.
4. **Motion Consistency ($S_{motion}$, 15%)**: Kinetic impact deceleration measured via Farneback Optical Flow.
5. **Road Layout Hazard ($H_{layout}$, 10%)**: Road speed corridor context (`highway` = 1.0, `tunnel` = 0.95, `intersection` = 0.75, `street` = 0.60).

---

## 4. Scientifically Valid Conclusions & System Boundaries

- **Visual Evidence Grounding**: SASA relies strictly on observable spatial, kinetic, and context metadata.
- **Zero Medical Claims**: SASA estimates **Visual Collision Severity**, not medical health outcomes or injury severity.
- **Zero Fabricated Accuracy**: Because public dataset metadata lacks explicit medical ground truth, no fake classification accuracy metrics are claimed.
