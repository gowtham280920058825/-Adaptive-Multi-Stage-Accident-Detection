# MODEL TRAINING & EVALUATION GUIDE

This guide details how to prepare datasets, train baseline and base-paper models, and execute ablation study evaluation benchmarks.

---

## 1. Dataset Preparation

The system automatically extracts key frames from `d:\New folder\dataset\real_videos\` based on `metadata-real.csv`.

To prepare or re-generate the dataset in YOLO format:
```cmd
python backend/training/prepare_dataset.py
```
This extracts accident frames and normal background frames into:
- `dataset/prepared/images/train`, `val`, `test`
- `dataset/prepared/labels/train`, `val`, `test`
- `dataset/prepared/dataset.yaml`

---

## 2. Training Baseline YOLO11

Run the baseline YOLO11 training script:
```cmd
python backend/training/train_yolo11.py
```
Weights will be saved under `experiments/baseline/` and `backend/models/yolov11_baseline.pt`.

---

## 3. Training Base Paper Model (YOLO11-AMF)

To train the improved base paper model incorporating MLLA, AFPN, and Focaler-IoU hyperparameter tuning:
```cmd
python backend/training/train_yolo11_amf.py
```
Weights will be saved under `experiments/base_paper/` and `backend/models/yolov11_amf.pt`.

---

## 4. Running 5-Stage Ablation Study & Evaluation

Execute the evaluation benchmark script:
```cmd
python backend/training/evaluate.py
```
This benchmarks 5 configurations:
1. Config A: Baseline YOLO11
2. Config B: Base Paper YOLO11-AMF
3. Config C: YOLO11-AMF + AMAVE
4. Config D: YOLO11-AMF + AMAVE + SASA
5. Config E: Full Proposed Framework (+AMAVE + SASA + SEDE)

Output JSON reports and plots will be written to `results/metrics/ablation_study_results.json` and `results/plots/ablation_precision_comparison.png`.
