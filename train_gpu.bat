@echo off
echo Starting GPU Model Training for YOLO11 Baseline & Base Paper YOLO11-AMF...
python backend/training/train_yolo11.py
python backend/training/train_yolo11_amf.py
pause
