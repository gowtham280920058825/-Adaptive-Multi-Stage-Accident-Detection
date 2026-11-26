import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import cv2
from backend.detection.yolov11_amf import YOLO11AMFDetector
from backend.config import YOLO_AMF_WEIGHTS

det = YOLO11AMFDetector(weights_path=YOLO_AMF_WEIGHTS, conf_thresh=0.20)
for f_idx in [20, 30, 34, 40, 50]:
    frame = cv2.imread(f"backend/training/0nixr_f{f_idx}.jpg")
    res = det.detect_frame(frame)
    boxes = [(round(r["confidence"], 2), [round(x,1) for x in r["box"]]) for r in res]
    print(f"F{f_idx}: {len(res)} dets -> {boxes}")
