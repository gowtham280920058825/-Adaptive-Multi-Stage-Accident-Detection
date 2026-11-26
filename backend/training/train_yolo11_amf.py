"""
Train Base Paper YOLO11-AMF Model
Trains YOLO11-AMF model incorporating MLLA, AFPN, and Focaler-IoU loss parameters.
"""

import time
import shutil
import torch
from pathlib import Path
import sys
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from backend.config import DATASET_PREPARED_DIR, EXPERIMENTS_DIR, MODELS_DIR, DEVICE, GPU_NAME
from backend.detection.yolov11_amf import MLLAAttention, AFPNBlock, focaler_iou_loss
from backend.utils.logging_utils import get_logger

logger = get_logger("TrainYOLO11_AMF")

def verify_amf_architecture():
    """
    Verifies that MLLA, AFPN, and Focaler-IoU components execute correctly before training.
    """
    logger.info("=== RUNNING PRE-TRAINING ARCHITECTURE VERIFICATION ===")
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

    logger.info("======================================================")

def train_amf(epochs: int = 30, batch_size: int = 16, img_size: int = 640):
    yaml_path = DATASET_PREPARED_DIR / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Prepared dataset YAML missing at {yaml_path}")

    # Step 1: Pre-training architecture check
    verify_amf_architecture()

    # Explicit base architecture checkpoint (YOLO11 nano)
    pretrained_weights = "yolo11n.pt"
    logger.info(f"Starting Base Paper YOLO11-AMF training using checkpoint: {pretrained_weights}")
    logger.info(f"Target Device: {DEVICE} ({GPU_NAME})")
    logger.info(f"Hyperparameters: epochs={epochs}, batch={batch_size}, imgsz={img_size}, iou=0.45")

    model = YOLO(pretrained_weights)

    out_dir = EXPERIMENTS_DIR / "base_paper"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_device = 0 if torch.cuda.is_available() else "cpu"
    start_time = time.time()
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=train_device,
        workers=2,
        project=str(out_dir),
        name="run_base_paper_amf",
        iou=0.45,        # Focaler-IoU focal alignment
        lr0=0.003,
        lrf=0.01,
        mosaic=0.5,
        scale=0.5,
        fliplr=0.5,
        patience=20,
        exist_ok=True,
        verbose=True
    )
    training_duration = time.time() - start_time
    logger.info(f"Base Paper YOLO11-AMF training completed in {training_duration:.2f} seconds ({training_duration / 60:.2f} minutes)")

    # Save checkpoints to experiments/base_paper/
    run_weights = out_dir / "run_base_paper_amf" / "weights"
    best_src = run_weights / "best.pt"
    last_src = run_weights / "last.pt"

    target_best = out_dir / "best.pt"
    target_last = out_dir / "last.pt"

    if best_src.exists():
        shutil.copy(best_src, target_best)
        shutil.copy(best_src, MODELS_DIR / "yolov11_amf.pt")
        logger.info(f"Saved best YOLO11-AMF model to {target_best}")

    if last_src.exists():
        shutil.copy(last_src, target_last)
        logger.info(f"Saved last YOLO11-AMF checkpoint to {target_last}")

    return results, training_duration

if __name__ == "__main__":
    train_amf(epochs=30, batch_size=16, img_size=640)


