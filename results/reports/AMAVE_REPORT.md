# AMAVE (Adaptive Multi-Stage Accident Verification Engine) REPORT

**Module Status**: **Completed**  
**Evaluation Scope**: 54 Held-Out Test Frames across 9 Test Videos (Exact 60-Video Split)  
**Metrics Output Path**: `results/metrics/amave_metrics.json`  
**Visual Evidence Path**: `results/reports/amave_visualization/`  

---

## 1. Executive Summary & Verification Impact

- **False-Alert Reduction**: **0.00%** reduction in false alarms (Raw FP: 2 $\rightarrow$ AMAVE FP: 2).
- **Correctly Verified Accidents**: **3**
- **Rejected False Events**: **25**
- **Average AMAVE Verification Latency**: **37.54 ms**

---

## 2. 5-Stage Ablation Study Performance Breakdown

| Configuration Stage | Precision | Recall | F1-Score | Accuracy | False-Positive Rate (FPR) | Avg Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Config A: YOLO11-AMF Only** | 0.7500 | 0.2222 | 0.3429 | 0.5741 | 0.0741 | 37.54 ms |
| **Config B: + Temporal Persistence** | 0.6667 | 0.1481 | 0.2424 | 0.5370 | 0.0741 | 37.54 ms |
| **Config C: + Motion Consistency** | 0.7500 | 0.1111 | 0.1935 | 0.5370 | 0.0370 | 37.54 ms |
| **Config D: + Collision Interaction** | 0.7500 | 0.1111 | 0.1935 | 0.5370 | 0.0370 | 37.54 ms |
| **Config E: Full AMAVE Engine** | **0.6000** | **0.1111** | **0.1875** | **0.5185** | **0.0741** | **37.54 ms** |

---

## 3. AMAVE Output Format Verification

Each candidate event evaluated by AMAVE produces standardized outputs:
- `detection_confidence`
- `temporal_persistence_score`
- `motion_consistency_score`
- `interaction_score`
- `fused_verification_score`
- `verification_status` (`VERIFIED_ACCIDENT` or `UNVERIFIED_EVENT`)
