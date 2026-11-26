# END-TO-END PIPELINE INTEGRATION AUDIT REPORT

**Active Framework Pipeline**: `VIDEO` $\rightarrow$ `YOLO11-AMF` $\rightarrow$ `AMAVE V2` $\rightarrow$ `SASA` $\rightarrow$ `SEDE`  
**Test Scope**: 9 Held-Out Test Videos (54 Frames)  

---

## 1. Video Sequence Processing Results

| Test Video ID | Total Frames | Pipeline Status | Verified Frames | Processing Time (FPS) | Annotated Output |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `2wUrP2wgXhs_00` | 6 | **UNVERIFIED / NORMAL** | 0 | 2.41s (2.49 FPS) | [`audit_annotated_2wUrP2wgXhs_00.mp4`](file:///D:/New folder/r&d/results/reports/end_to_end_tests/audit_annotated_2wUrP2wgXhs_00.mp4 ) |
| `eQ8yX8V88l8_00` | 6 | **VERIFIED_ACCIDENT** | 1 | 0.16s (38.63 FPS) | [`audit_annotated_eQ8yX8V88l8_00.mp4`](file:///D:/New folder/r&d/results/reports/end_to_end_tests/audit_annotated_eQ8yX8V88l8_00.mp4 ) |
| `fbkRjY737QE_00` | 6 | **UNVERIFIED / NORMAL** | 0 | 0.14s (42.82 FPS) | [`audit_annotated_fbkRjY737QE_00.mp4`](file:///D:/New folder/r&d/results/reports/end_to_end_tests/audit_annotated_fbkRjY737QE_00.mp4 ) |
| `JZcTEwl1Od4_00` | 6 | **VERIFIED_ACCIDENT** | 2 | 0.52s (11.59 FPS) | [`audit_annotated_JZcTEwl1Od4_00.mp4`](file:///D:/New folder/r&d/results/reports/end_to_end_tests/audit_annotated_JZcTEwl1Od4_00.mp4 ) |
| `n1da-gnbUBU_00` | 6 | **UNVERIFIED / NORMAL** | 0 | 0.14s (42.19 FPS) | [`audit_annotated_n1da-gnbUBU_00.mp4`](file:///D:/New folder/r&d/results/reports/end_to_end_tests/audit_annotated_n1da-gnbUBU_00.mp4 ) |
| `pA6q7Q2EQrw_32_00` | 6 | **UNVERIFIED / NORMAL** | 0 | 0.13s (44.7 FPS) | [`audit_annotated_pA6q7Q2EQrw_32_00.mp4`](file:///D:/New folder/r&d/results/reports/end_to_end_tests/audit_annotated_pA6q7Q2EQrw_32_00.mp4 ) |
| `QSPGXxzUXVs_00` | 6 | **UNVERIFIED / NORMAL** | 0 | 0.2s (29.72 FPS) | [`audit_annotated_QSPGXxzUXVs_00.mp4`](file:///D:/New folder/r&d/results/reports/end_to_end_tests/audit_annotated_QSPGXxzUXVs_00.mp4 ) |
| `SBTD7z4A1y8_00` | 6 | **VERIFIED_ACCIDENT** | 4 | 1.48s (4.05 FPS) | [`audit_annotated_SBTD7z4A1y8_00.mp4`](file:///D:/New folder/r&d/results/reports/end_to_end_tests/audit_annotated_SBTD7z4A1y8_00.mp4 ) |
| `video26_00` | 6 | **UNVERIFIED / NORMAL** | 0 | 0.58s (10.32 FPS) | [`audit_annotated_video26_00.mp4`](file:///D:/New folder/r&d/results/reports/end_to_end_tests/audit_annotated_video26_00.mp4 ) |

---

## 2. Gating & Verification Integrity Checks

1. **AMAVE V2 $\rightarrow$ SASA Gating**:
   - For all unverified frames ($47/54$), SASA correctly returned `sasa_status`: `NOT_EXECUTED`, `severity`: `N/A`. Zero false executions.
2. **SASA $\rightarrow$ SEDE Gating**:
   - For all unverified frames ($47/54$), SEDE correctly returned `sede_status`: `NOT_EXECUTED`, `recommended_action`: `NO_ACTION`. Zero false executions.
3. **Execution Latency**:
   - Detector + AMAVE V2 + SASA + SEDE combined latency: **~38.97 ms** (~25.7 FPS execution speed).
