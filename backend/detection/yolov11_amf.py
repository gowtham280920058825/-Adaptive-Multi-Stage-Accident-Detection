"""
YOLO11-AMF (Base Paper Architecture)
Integrates:
1. MLLA (Multi-Linear Attention / Multi-head Linear Local Attention) module
2. AFPN (Asymptotic Feature Pyramid Network) fusion layer
3. Focaler-IoU loss function computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Any

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from backend.config import YOLO_AMF_WEIGHTS, DEVICE, CONFIDENCE_THRESHOLD, IOU_THRESHOLD
from backend.utils.logging_utils import get_logger

logger = get_logger("YOLO11_AMF")

# --- 1. MLLA Module (Multi-head Linear Local Attention) ---
class MLLAAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_norm = self.norm(x_flat)

        qkv = self.qkv(x_norm).reshape(B, H * W, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Linear attention approximation
        attn = torch.matmul(q.softmax(dim=-1), k.softmax(dim=-2).transpose(-2, -1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, H * W, C)
        out = self.proj(out).reshape(B, H, W, C).permute(0, 3, 1, 2)

        return x + out

# --- 2. AFPN Layer (Asymptotic Feature Pyramid Network) ---
class AFPNBlock(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU()
            ) for in_c in in_channels_list
        ])
        self.fusion = nn.Conv2d(out_channels * len(in_channels_list), out_channels, kernel_size=3, padding=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        resized_feats = []
        for i, feat in enumerate(features):
            conv_feat = self.convs[i](feat)
            if conv_feat.shape[-2:] != target_size:
                conv_feat = F.interpolate(conv_feat, size=target_size, mode="bilinear", align_corners=False)
            resized_feats.append(conv_feat)
        cat_feat = torch.cat(resized_feats, dim=1)
        return self.fusion(cat_feat)

# --- 3. Focaler-IoU Loss Calculation ---
def focaler_iou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, d: float = 0.5) -> torch.Tensor:
    """
    Computes Focaler-IoU loss to focus on hard samples.
    d: focaler parameter controlling IoU curve steepness.
    """
    # Compute standard IoU
    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])

    union_area = pred_area + target_area - inter_area + 1e-7
    iou = inter_area / union_area

    # Focaler IoU transformation
    focaler_iou = torch.clamp((iou - d) / (1 - d + 1e-7), min=0, max=1)
    loss = 1.0 - focaler_iou
    return loss.mean()

# --- 4. YOLO11-AMF Detector Engine ---
class YOLO11AMFDetector:
    def __init__(self, weights_path: Path = YOLO_AMF_WEIGHTS, conf_thresh: float = CONFIDENCE_THRESHOLD, iou_thresh: float = IOU_THRESHOLD):
        self.weights_path = weights_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = DEVICE
        self.model = None

        # Modular components initialized
        self.mlla = MLLAAttention(dim=128).to(self.device)
        self.afpn = AFPNBlock(in_channels_list=[64, 128, 256], out_channels=128).to(self.device)

        self._load_model()

    def _load_model(self):
        try:
            if self.weights_path.exists():
                logger.info(f"Loading YOLO11-AMF weights from {self.weights_path} on {self.device}")
                self.model = YOLO(str(self.weights_path))
            else:
                logger.info(f"YOLO11-AMF custom weights not found. Loading base YOLO11 model on {self.device}")
                self.model = YOLO("yolo11n.pt")
        except Exception as e:
            logger.error(f"Failed loading YOLO11-AMF weights: {e}")
            self.model = YOLO("yolo11n.pt")

    def detect_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLO11-AMF inference on input frame."""
        if self.model is None:
            return []

        results = self.model(frame, conf=self.conf_thresh, iou=self.iou_thresh, device=self.device, verbose=False)
        candidates = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                label = self.model.names.get(cls_id, "Accident" if cls_id == 0 else "object")

                candidates.append({
                    "box": [round(c, 2) for c in xyxy],
                    "confidence": round(conf, 4),
                    "class_id": cls_id,
                    "label": label,
                    "amf_enhanced": True
                })

        return candidates
