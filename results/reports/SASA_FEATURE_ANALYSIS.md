# SASA FEATURE CONTRIBUTION & ABLATION STUDY REPORT

**Scope**: Multi-Feature Contribution Analysis across 5 Configurations on Held-Out Test Set  

---

## 1. Feature Contribution Ablation Matrix

| Configuration Stage | Included Features | Mean Score | Std Dev | LOW | MODERATE | HIGH | CRITICAL | Avg Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Config A: Type Only** | $H_{type}$ | 0.8857 | 0.1705 | 0 | 1 | 0 | 6 | 0.08 ms |
| **Config B: + Rollover** | $H_{type} + F_{roll}$ | 0.4871 | 0.0938 | 1 | 6 | 0 | 0 | 0.15 ms |
| **Config C: + Vehicles** | $+ N_{veh}$ | 0.4436 | 0.0836 | 1 | 6 | 0 | 0 | 0.23 ms |
| **Config D: + Motion** | $+ S_{motion}$ | 0.4686 | 0.0798 | 1 | 6 | 0 | 0 | 0.30 ms |
| **Config E: Full SASA** | $+ H_{layout}$ (Composite) | **0.4993** | **0.0727** | **0** | **7** | **0** | **0** | **0.38 ms** |

---

## 2. Feature Sensitivity Analysis

1. **Collision Geometry ($H_{type}$)**: Provides the baseline risk stratification (e.g. Head-on vs Rear-end).
2. **Rollover Flag ($F_{roll}$)**: Shifts event categorization directly into `HIGH` or `CRITICAL` due to structural crash severity.
3. **Vehicle Involvement ($N_{veh}$)**: Increases score linearly for multi-vehicle collisions.
4. **Optical Flow Motion ($S_{motion}$)**: Adds dynamic kinetic energy dissipation evidence.
