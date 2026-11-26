#!/bin/bash
echo "Starting GPU Training..."
python3 backend/training/train_yolo11.py
python3 backend/training/train_yolo11_amf.py
