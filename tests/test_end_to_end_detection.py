"""
Automated End-to-End Test Harness for Accident Detection Pipeline
Evaluates fixed set of real-world videos against verified ground truth.
Pipeline: Video -> YOLO11-AMF -> AMAVE V2 -> SASA -> SEDE -> Final Decision.
Calculates authentic confusion matrix without fabrication or manual entry.
"""

import sys
import os
import json
import time
from pathlib import Path
import cv2
import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.append(str(WORKSPACE))

from backend.config import YOLO_AMF_WEIGHTS, DEVICE, CONFIDENCE_THRESHOLD, MIN_VERIFIED_FRAMES
from backend.detection.yolov11_amf import YOLO11AMFDetector
from backend.detection.video_processor import VideoProcessor
from backend.detection.temporal_engine import TemporalEngine
from backend.amave.engine import AMAVEEngine
from backend.sasa.analyzer import SASAAnalyzer
from backend.sede.decision_engine import SEDEDecisionEngine

# Test suite containing diverse real-world test cases:
# 1. Clear daytime/morning accident: _9tp71S2Gwg_00.mp4
# 2. Difficult highway accident: __WFqm4i3vE_00.mp4
# 3. Night CCTV accident: _0RDZgCKPGo_00.mp4
# 4. Urban night accident: 0NixrF6Mt00_0_00.mp4
# 5. Signalized intersection accident: -6SQSDj8cYU_00.mp4
# 6. Low-light highway accident: Z4kg2Ev3vhk_00.mp4
# 7. Difficult low-light accident: nAXTthLfgtI_00.mp4
# 8. Normal clear traffic: normal_traffic_clear.mp4
# 9. Normal urban traffic: normal_traffic_urban.mp4
# 10. Normal highway test: test_normal_traffic.mp4

TEST_CASES = [
    {
        "video_name": "_9tp71S2Gwg_00.mp4",
        "ground_truth": "ACCIDENT",
        "category": "Clear Morning/Daytime Highway Accident"
    },
    {
        "video_name": "__WFqm4i3vE_00.mp4",
        "ground_truth": "ACCIDENT",
        "category": "High-Speed Highway Collision"
    },
    {
        "video_name": "_0RDZgCKPGo_00.mp4",
        "ground_truth": "ACCIDENT",
        "category": "Night CCTV Highway Accident"
    },
    {
        "video_name": "0NixrF6Mt00_0_00.mp4",
        "ground_truth": "ACCIDENT",
        "category": "Urban Intersection Crash"
    },
    {
        "video_name": "-6SQSDj8cYU_00.mp4",
        "ground_truth": "ACCIDENT",
        "category": "Signalized Intersection T-Bone"
    },
    {
        "video_name": "Z4kg2Ev3vhk_00.mp4",
        "ground_truth": "ACCIDENT",
        "category": "Night Highway Rear-End"
    },
    {
        "video_name": "nAXTthLfgtI_00.mp4",
        "ground_truth": "ACCIDENT",
        "category": "Low-Light Multi-Vehicle Collision"
    },
    {
        "video_name": "normal_traffic_clear.mp4",
        "ground_truth": "NORMAL",
        "category": "Normal Traffic Daytime Flow"
    },
    {
        "video_name": "normal_traffic_urban.mp4",
        "ground_truth": "NORMAL",
        "category": "Normal Urban Arterial Traffic"
    },
    {
        "video_name": "test_normal_traffic.mp4",
        "ground_truth": "NORMAL",
        "category": "Normal Highway Surveillance"
    }
]

def find_video_path(vname):
    # Search in dataset/real_videos, evaluation_videos, dataset root, and results/temp_uploads
    candidates = [
        WORKSPACE / "dataset" / "real_videos" / vname,
        WORKSPACE / "dataset" / "evaluation_videos" / "accident" / vname,
        WORKSPACE / "dataset" / "evaluation_videos" / "non_accident" / vname,
        WORKSPACE / "dataset" / vname,
        WORKSPACE / "results" / "temp_uploads" / vname,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def run_test_harness():
    print("=" * 80)
    print("END-TO-END AUTOMATED TEST HARNESS EXECUTION")
    print(f"Model Checkpoint: {YOLO_AMF_WEIGHTS}")
    print(f"Device: {DEVICE}")
    print(f"Candidate Proposal Threshold: {CONFIDENCE_THRESHOLD}")
    print("=" * 80)

    detector = YOLO11AMFDetector(weights_path=YOLO_AMF_WEIGHTS, conf_thresh=CONFIDENCE_THRESHOLD)
    amave_engine = AMAVEEngine()
    sasa_analyzer = SASAAnalyzer()
    sede_engine = SEDEDecisionEngine()

    results = []
    tp, tn, fp, fn = 0, 0, 0, 0

    for idx, tc in enumerate(TEST_CASES, 1):
        vname = tc["video_name"]
        gt = tc["ground_truth"]
        cat = tc["category"]

        vpath = find_video_path(vname)
        if not vpath:
            print(f"[{idx}/{len(TEST_CASES)}] ERROR: Video not found: {vname}")
            continue

        print(f"\n[{idx}/{len(TEST_CASES)}] Testing: {vname} ({cat}) | Ground Truth: {gt}")

        vp = VideoProcessor(vpath)
        temp_engine = TemporalEngine(window_size=10)
        frames_cache = []
        all_detections = []
        frame_amave_results = []
        max_detection_confidence = 0.0
        candidate_frames_count = 0

        t0 = time.time()
        for f_idx, timestamp, frame in vp.extract_frames(stride=2):
            frames_cache.append(frame)
            dets = detector.detect_frame(frame)
            temp_engine.push_frame_detections(f_idx, timestamp, dets)
            all_detections.extend(dets)

            if dets:
                candidate_frames_count += 1
                c_max = max([d["confidence"] for d in dets], default=0.0)
                max_detection_confidence = max(max_detection_confidence, c_max)

            h_win = list(temp_engine.history)
            f_win = frames_cache[max(0, len(frames_cache) - 2):]
            amave_res = amave_engine.verify_event(dets, h_win, f_win)
            frame_amave_results.append(amave_res)

        vp.release()
        elapsed = time.time() - t0

        verified_windows = [res for res in frame_amave_results if res.get("verified", False)]
        is_accident_verified = (len(verified_windows) >= MIN_VERIFIED_FRAMES)
        final_prediction = "ACCIDENT" if is_accident_verified else "NORMAL"

        if frame_amave_results:
            peak_amave = max(frame_amave_results, key=lambda x: x.get("fused_verification_score", 0.0))
            peak_fused = peak_amave.get("fused_verification_score", 0.0)
            peak_status = "VERIFIED" if is_accident_verified else peak_amave.get("verification_status", "UNVERIFIED")
            peak_reason = peak_amave.get("reason", "")
        else:
            peak_fused = 0.0
            peak_status = "UNVERIFIED"
            peak_reason = "No frames evaluated"

        # SASA / SEDE execution if verified
        if is_accident_verified:
            peak_dets = [d for d in all_detections if d.get("confidence", 0.0) >= CONFIDENCE_THRESHOLD]
            sasa_res = sasa_analyzer.analyze_severity(peak_amave, peak_dets)
            sede_res = sede_engine.make_decision(peak_amave, sasa_res, vname)
            sasa_sev = sasa_res.get("severity", "MODERATE")
            sede_act = sede_res.get("recommended_action", "DISPATCH_EMERGENCY")
        else:
            sasa_sev = "N/A"
            sede_act = "NO_ACTION"

        # Evaluate against Ground Truth
        if gt == "ACCIDENT":
            if final_prediction == "ACCIDENT":
                tp += 1
                eval_status = "TRUE_POSITIVE (CORRECT)"
            else:
                fn += 1
                eval_status = "FALSE_NEGATIVE (MISSED)"
        else:
            if final_prediction == "NORMAL":
                tn += 1
                eval_status = "TRUE_NEGATIVE (CORRECT)"
            else:
                fp += 1
                eval_status = "FALSE_POSITIVE (INCORRECT)"

        row = {
            "video_name": vname,
            "category": cat,
            "ground_truth": gt,
            "yolo_max_confidence": round(float(max_detection_confidence), 4),
            "yolo_detection_frames": candidate_frames_count,
            "yolo_total_candidates": len(all_detections),
            "amave_score": round(float(peak_fused), 4),
            "amave_status": peak_status,
            "final_prediction": final_prediction,
            "eval_status": eval_status,
            "sasa_severity": sasa_sev,
            "sede_action": sede_act,
            "elapsed_sec": round(elapsed, 2)
        }
        results.append(row)

        print(f"  Result: Final={final_prediction} (GT={gt}) -> {eval_status}")
        print(f"  YOLO: MaxConf={max_detection_confidence*100:.1f}%, Frames={candidate_frames_count}, Boxes={len(all_detections)}")
        print(f"  AMAVE: Score={peak_fused:.4f}, Status={peak_status}, Reason={peak_reason[:60]}")

    # Metrics calculation
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("\n" + "=" * 80)
    print("TEST HARNESS EVALUATION MATRIX SUMMARY")
    print("=" * 80)
    print(f"Total Evaluated Videos: {total}")
    print(f"True Positives (TP):  {tp}")
    print(f"True Negatives (TN):  {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Event Accuracy:      {acc*100:.2f}%")
    print(f"Event Precision:     {prec*100:.2f}%")
    print(f"Event Recall:        {rec*100:.2f}%")
    print(f"Event F1-Score:      {f1*100:.2f}%")
    print(f"Event FPR:           {fpr*100:.2f}%")

    out_file = WORKSPACE / "results" / "metrics" / "test_harness_evaluation.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "YOLO11-AMF",
            "weights": str(YOLO_AMF_WEIGHTS),
            "summary": {
                "total": total,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "fpr": round(fpr, 4)
            },
            "cases": results
        }, f, indent=2)

    print(f"\nSaved test results to {out_file}")
    return results, (tp, tn, fp, fn, acc, prec, rec, f1, fpr)

if __name__ == "__main__":
    run_test_harness()
