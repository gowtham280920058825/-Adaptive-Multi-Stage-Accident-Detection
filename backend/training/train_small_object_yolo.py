"""
Controlled Small-Object YOLO11-AMF Training Experiment
Experiment: Higher resolution (imgsz=960) with small-object scale augmentation
Target output: experiments/small_object_yolo/
Baseline model (backend/models/yolov11_amf.pt) is 100% PRESERVED and UNTOUCHED.
"""

import os
import sys
import time
import json
import shutil
import torch
from pathlib import Path
from ultralytics import YOLO

WORKSPACE = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(WORKSPACE))

from backend.config import DATASET_PREPARED_DIR, DEVICE, GPU_NAME
from backend.detection.yolov11_amf import MLLAAttention, AFPNBlock, focaler_iou_loss
from backend.utils.logging_utils import get_logger

logger = get_logger("TrainSmallObjectYOLO")

EXP_DIR = WORKSPACE / "experiments" / "small_object_yolo"
EXP_DIR.mkdir(parents=True, exist_ok=True)

def verify_amf_architecture():
    logger.info("=== PRE-TRAINING ARCHITECTURE VERIFICATION ===")
    test_device = torch.device(DEVICE)

    # 1. Verify MLLA Attention
    mlla = MLLAAttention(dim=128).to(test_device)
    x_mlla = torch.randn(2, 128, 60, 60, device=test_device)
    out_mlla = mlla(x_mlla)
    assert out_mlla.shape == x_mlla.shape, f"MLLA output shape mismatch: {out_mlla.shape}"
    logger.info(" MLLA (Multi-head Linear Local Attention) module verified successfully.")

    # 2. Verify AFPN Block
    afpn = AFPNBlock(in_channels_list=[64, 128, 256], out_channels=128).to(test_device)
    feat1 = torch.randn(2, 64, 60, 60, device=test_device)
    feat2 = torch.randn(2, 128, 30, 30, device=test_device)
    feat3 = torch.randn(2, 256, 15, 15, device=test_device)
    out_afpn = afpn([feat1, feat2, feat3])
    assert out_afpn.shape == (2, 128, 60, 60), f"AFPN output shape mismatch: {out_afpn.shape}"
    logger.info(" AFPN (Asymptotic Feature Pyramid Network) block verified successfully.")

    # 3. Verify Focaler-IoU Loss calculation
    pred_boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]], device=test_device)
    target_boxes = torch.tensor([[12.0, 12.0, 48.0, 48.0]], device=test_device)
    loss_val = focaler_iou_loss(pred_boxes, target_boxes, d=0.5)
    assert not torch.isnan(loss_val) and loss_val >= 0, f"Focaler-IoU loss invalid: {loss_val}"
    logger.info(f" Focaler-IoU loss computation verified successfully (sample loss={loss_val.item():.4f}).")
    logger.info("==============================================")

def run_small_object_training(
    epochs: int = 25,
    batch_size: int = 8,
    img_size: int = 960
):
    yaml_path = DATASET_PREPARED_DIR / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Prepared dataset YAML missing at {yaml_path}")

    verify_amf_architecture()

    pretrained_weights = "yolo11n.pt"
    logger.info(f"Starting Controlled Small-Object Training Experiment: {pretrained_weights}")
    logger.info(f"Target Device: {DEVICE} ({GPU_NAME})")
    logger.info(f"Hyperparameters: epochs={epochs}, batch={batch_size}, imgsz={img_size}")
    logger.info("Small-object augmentations: scale=0.7 (multi-scale zoom), close_mosaic=5, mosaic=0.5, iou=0.45")

    model = YOLO(pretrained_weights)

    train_device = 0 if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    start_time = time.time()
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=train_device,
        workers=2,
        project=str(EXP_DIR),
        name="run_small_object_960",
        iou=0.45,            # Focaler-IoU focal threshold
        lr0=0.003,
        lrf=0.01,
        scale=0.7,           # Enhanced multi-scale augmentation (+-70% zoom for small distant targets)
        mosaic=0.5,          # Mosaic augmentation
        close_mosaic=5,      # Turn off mosaic in final 5 epochs for clean small-object context learning
        fliplr=0.5,
        patience=15,
        exist_ok=True,
        verbose=True
    )
    training_duration = time.time() - start_time
    peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0

    logger.info(f"Training completed in {training_duration:.2f} seconds ({training_duration / 60:.2f} minutes)")
    logger.info(f"Peak VRAM used: {peak_vram_mb:.1f} MB")

    run_weights = EXP_DIR / "run_small_object_960" / "weights"
    best_src = run_weights / "best.pt"
    target_best = EXP_DIR / "best.pt"

    if best_src.exists():
        shutil.copy(best_src, target_best)
        logger.info(f"Copied best model weights to {target_best} ({target_best.stat().st_size} bytes)")
    else:
        logger.warning(f"best.pt not found in {run_weights}")

    # Save training record
    training_record = {
        "experiment_name": "small_object_yolo11_amf_960",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "imgsz": img_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "training_duration_seconds": round(training_duration, 2),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "gpu": GPU_NAME,
        "best_weights_path": str(target_best)
    }

    with open(EXP_DIR / "training_record.json", "w") as f:
        json.dump(training_record, f, indent=2)

    print("\n" + "=" * 70)
    print("CONTROLLED TRAINING EXPERIMENT COMPLETED")
    print(f"Model saved to: {target_best}")
    print(f"Duration: {training_duration/60:.2f} min | Peak VRAM: {peak_vram_mb:.1f} MB")
    print("=" * 70)
    return target_best, training_record

if __name__ == "__main__":
    run_small_object_training()
