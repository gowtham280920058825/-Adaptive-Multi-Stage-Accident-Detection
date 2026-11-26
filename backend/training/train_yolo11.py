"""
Corrective Baseline YOLO11 Training Module
Initializes from official yolo11n.pt checkpoint, trains for 20 epochs on GPU.
Saves checkpoints to experiments/baseline_corrective/best.pt and last.pt
"""

import time
import shutil
from pathlib import Path
import sys
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from backend.config import DATASET_PREPARED_DIR, EXPERIMENTS_DIR, MODELS_DIR, DEVICE, GPU_NAME
from backend.utils.logging_utils import get_logger

logger = get_logger("TrainBaselineYOLO11Corrective")

def train_baseline_corrective(epochs: int = 20, batch_size: int = 8, img_size: int = 480):
    yaml_path = DATASET_PREPARED_DIR / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Prepared dataset YAML missing at {yaml_path}")

    pretrained_weights = "yolo11n.pt"
    logger.info(f"Initializing YOLO11 baseline training using explicit checkpoint: {pretrained_weights}")
    logger.info(f"Target Device: {DEVICE} ({GPU_NAME})")
    logger.info(f"Hyperparameters: epochs={epochs}, batch={batch_size}, imgsz={img_size}")

    model = YOLO(pretrained_weights)

    out_dir = EXPERIMENTS_DIR / "baseline_corrective"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=DEVICE,
        workers=2,
        project=str(out_dir),
        name="run_corrective",
        exist_ok=True,
        verbose=True
    )
    training_duration = time.time() - start_time
    logger.info(f"Training completed in {training_duration:.2f} seconds ({training_duration / 60:.2f} minutes)")

    # Copy best.pt and last.pt directly to experiments/baseline_corrective/
    run_weights = out_dir / "run_corrective" / "weights"
    best_src = run_weights / "best.pt"
    last_src = run_weights / "last.pt"

    target_best = out_dir / "best.pt"
    target_last = out_dir / "last.pt"

    if best_src.exists():
        shutil.copy(best_src, target_best)
        shutil.copy(best_src, MODELS_DIR / "yolov11_baseline.pt")
        logger.info(f"Saved best model to {target_best}")

    if last_src.exists():
        shutil.copy(last_src, target_last)
        logger.info(f"Saved last checkpoint to {target_last}")

    return results, training_duration

if __name__ == "__main__":
    train_baseline_corrective(epochs=20, batch_size=8, img_size=480)

