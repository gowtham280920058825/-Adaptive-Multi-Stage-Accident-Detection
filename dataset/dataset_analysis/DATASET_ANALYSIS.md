# DATASET ANALYSIS & VERIFICATION REPORT

**Dataset Location**: `dataset/`  
**Total Real Videos**: 2027  
**Total Real Metadata Records**: 2027  
**Total Synthetic Metadata Records**: 2211  

---

## 1. Collision Category Distribution

| Collision Type | Count | Percentage |
| :--- | :--- | :--- |
| single | 680 | 33.55% |
| t-bone | 657 | 32.41% |
| rear-end | 328 | 16.18% |
| sideswipe | 245 | 12.09% |
| head-on | 117 | 5.77% |

---

## 2. Environmental & Context Features

### Weather Distribution
- **normal**: 1651 (81.5%)
- **rain**: 264 (13.0%)
- **snow**: 112 (5.5%)

### Day / Night Distribution
- **day**: 1376 (67.9%)
- **night**: 651 (32.1%)

### Scene Layout Distribution
- **highway**: 937 (46.2%)
- **signalized_intersection**: 703 (34.7%)
- **grade_separated_intersection**: 156 (7.7%)
- **simple_intersection**: 100 (4.9%)
- **city_street**: 81 (4.0%)
- **tunnel**: 25 (1.2%)
- **parking_lot**: 17 (0.8%)
- **roundabout**: 8 (0.4%)

---

## 3. Video Statistics

- **Average Video Duration**: 21.85 seconds
- **Average Frame Count**: 377.4 frames
- **In-Distribution Train/Test Split**: {'test': 1520, 'train': 507}
- **Geo-Aware Train/Test Split**: {'test': 1573, 'train': 454}

---

## 4. Sample Bounding Box Verification Results

Verified **10** representative frames from real traffic videos. Sample rendered images saved in `results/reports/dataset_sample_verification/`.

| Video File | Accident Type | Target Frame | Rendered Sample |
| :--- | :--- | :--- | :--- |
| `real_videos/SBTD7z4A1y8_00.mp4` | head-on | 47 | `sample_SBTD7z4A1y8_00_frame47.jpg` |
| `real_videos/6hJ8mYLUfE8_00.mp4` | head-on | 273 | `sample_6hJ8mYLUfE8_00_frame273.jpg` |
| `real_videos/CkI203kbHQ8_11_00.mp4` | rear-end | 12 | `sample_CkI203kbHQ8_11_00_frame12.jpg` |
| `real_videos/XPErIB4xgYg_00.mp4` | rear-end | 108 | `sample_XPErIB4xgYg_00_frame108.jpg` |
| `real_videos/2wUrP2wgXhs_00.mp4` | sideswipe | 62 | `sample_2wUrP2wgXhs_00_frame62.jpg` |
| `real_videos/7CO_LqgfiFg_00.mp4` | sideswipe | 37 | `sample_7CO_LqgfiFg_00_frame37.jpg` |
| `real_videos/UcJKeuDYOzc_12_00.mp4` | single | 38 | `sample_UcJKeuDYOzc_12_00_frame38.jpg` |
| `real_videos/aERiIg9Dts4_00.mp4` | single | 78 | `sample_aERiIg9Dts4_00_frame78.jpg` |
| `real_videos/russia13_00.mp4` | t-bone | 65 | `sample_russia13_00_frame65.jpg` |
| `real_videos/4Bde0lYQ4mg_00.mp4` | t-bone | 44 | `sample_4Bde0lYQ4mg_00_frame44.jpg` |
