# AMAVE DIAGNOSTIC & REFINEMENT REPORT (AMAVE_V2)

**Evaluation Scope**: Validation-Only Hyperparameter Selection $\rightarrow$ Single-Pass Test Evaluation  
**Validation Data**: 48 Validation Frames (9 Validation Videos)  
**Held-Out Test Data**: 54 Test Frames (9 Test Videos)  
**Refined Metrics Output**: `results/metrics/amave_refined_metrics.json`  
**Distribution Plot Output**: `results/reports/amave_refinement/amave_score_distributions.png`  

---

## 1. Diagnostic Analysis: Why AMAVE_V1 Suppressed Recall

1. **Overly Strict Decision Threshold ($\\tau = 0.50$)**:
   In AMAVE_V1, setting a fixed threshold of 0.50 combined with fixed equal weights caused valid accident candidate events with moderate detection confidence ($0.15 - 0.30$) to fall just below the decision threshold.
2. **Temporal Window Constraint**:
   In low-frame-rate test videos, short accident bursts (1-2 frames) scored lower persistence ratios ($0.20 - 0.40$), severely dragging down the fused score when weighted at 30%.
3. **Refinement Remedy (Validation-Selected)**:
   By conducting grid search exclusively on **Validation Data**, the optimal fusion configuration was found:
   - **Weight Allocation**: **0.50 Detection** / **0.20 Persistence** / **0.15 Motion** / **0.15 Interaction**
   - **Verification Threshold**: **$\\tau = 0.30$**

---

## 2. Validation Set Grid Search Results

- **Best Validation F1-Score**: **0.3333** (Precision: 0.6667, Recall: 0.2222)
- **Optimal Weights**: $w_1 = 0.50, w_2 = 0.20, w_3 = 0.15, w_4 = 0.15$
- **Optimal Threshold**: $\\tau = 0.30$

---

## 3. Comparative Results: Original AMAVE_V1 vs Refined AMAVE_V2 on Held-Out Test Set

| Experiment Stage | Precision | Recall | F1-Score | Accuracy | False-Positive Rate (FPR) | False-Alert Reduction |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLO11-AMF Detector Only** | 0.7500 | 0.2222 | 0.3429 | 0.5741 | 0.0741 | Baseline |
| **Original AMAVE_V1** | 0.6000 | 0.1111 | 0.1875 | 0.5185 | 0.0741 | 0.0% |
| **Refined AMAVE_V2 (Val-Selected)** | **0.7143** | **0.1852** | **0.2941** | **0.5556** | **0.0741** | **0.00%** |

---

## 4. Full Ablation Study of AMAVE_V2 on Held-Out Test Set

| Configuration Stage | Precision | Recall | F1-Score | Accuracy | FPR | Avg Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Config A: Detector Only** | 0.7500 | 0.2222 | 0.3429 | 0.5741 | 0.0741 | 38.97 ms |
| **Config B: + Temporal Persistence** | 0.7500 | 0.2222 | 0.3429 | 0.5741 | 0.0741 | 38.97 ms |
| **Config C: + Motion Consistency** | 0.8000 | 0.1481 | 0.2500 | 0.5556 | 0.0370 | 38.97 ms |
| **Config D: + Collision Interaction** | 0.8000 | 0.1481 | 0.2500 | 0.5556 | 0.0370 | 38.97 ms |
| **Config E: Full AMAVE_V2** | **0.7143** | **0.1852** | **0.2941** | **0.5556** | **0.0741** | **38.97 ms** |
