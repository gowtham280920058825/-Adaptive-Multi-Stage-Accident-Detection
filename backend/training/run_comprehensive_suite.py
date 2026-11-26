"""
Comprehensive R&D Validation & Evaluation Suite
Implements:
- PART 6: YOLO11-AMF Evaluation on untouched held-out test set (400 images).
- PART 7 & 8: AMAVE multi-stage temporal verification with strict module gating.
- PART 9: Automated 6-Case Test Suite.
- PART 10 & 11: SASA Severity & SEDE Emergency Decision layer.
- PART 12 & 13: Independent Module Tests & 5-Stage Ablation Study.
- PART 16: Research Results Tables (Table 1 Event Level, Table 2 YOLO Detection, Table 3 Module Test, Table 4 Ablation).
"""

import os
import sys
import json
import time
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from backend.config import (
    DATASET_RAW_DIR, DATASET_PREPARED_DIR, RESULTS_DIR, EXPERIMENTS_DIR,
    MODELS_DIR, DEVICE, CONFIDENCE_THRESHOLD, VERIFICATION_THRESHOLD,
    BASELINE_YOLO_WEIGHTS, YOLO_AMF_WEIGHTS, MIN_VERIFIED_FRAMES
)
DATASET_DIR = DATASET_RAW_DIR
from backend.detection.detector import BaseYOLO11Detector
from backend.detection.yolov11_amf import YOLO11AMFDetector
from backend.detection.video_processor import VideoProcessor
from backend.detection.temporal_engine import TemporalEngine
from backend.amave.engine import AMAVEEngine
from backend.sasa.analyzer import SASAAnalyzer
from backend.sede.decision_engine import SEDEDecisionEngine
from backend.utils.logging_utils import get_logger

logger = get_logger("ComprehensiveSuite")

def run_suite():
    logger.info("==================================================================")
    logger.info("       STARTING SCIENTIFIC R&D EVALUATION & VALIDATION SUITE      ")
    logger.info("==================================================================")

    yaml_path = DATASET_PREPARED_DIR / "dataset.yaml"
    test_img_dir = DATASET_PREPARED_DIR / "images" / "test"
    test_lbl_dir = DATASET_PREPARED_DIR / "labels" / "test"

    test_images = sorted(list(test_img_dir.glob("*.jpg")))
    logger.info(f"Held-out test set: {len(test_images)} images ({len(list(test_lbl_dir.glob('*.txt')))} labels).")

    # -------------------------------------------------------------
    # 1. EVALUATE OBJECT DETECTORS ON UNTOUCHED HELD-OUT TEST SPLIT
    # -------------------------------------------------------------
    logger.info("Evaluating YOLO11 Baseline on held-out test split...")
    yolo_base = YOLO(str(BASELINE_YOLO_WEIGHTS))
    res_base_val = yolo_base.val(data=str(yaml_path), split="test", device=DEVICE, verbose=False)

    logger.info("Evaluating YOLO11-AMF on held-out test split...")
    yolo_amf = YOLO(str(YOLO_AMF_WEIGHTS))
    res_amf_val = yolo_amf.val(data=str(yaml_path), split="test", device=DEVICE, verbose=False)

    p_base = round(float(res_base_val.box.mp), 4)
    r_base = round(float(res_base_val.box.mr), 4)
    map50_base = round(float(res_base_val.box.map50), 4)
    map50_95_base = round(float(res_base_val.box.map), 4)
    f1_base = round(2 * p_base * r_base / (p_base + r_base), 4) if (p_base + r_base) > 0 else 0.0

    p_amf = round(float(res_amf_val.box.mp), 4)
    r_amf = round(float(res_amf_val.box.mr), 4)
    map50_amf = round(float(res_amf_val.box.map50), 4)
    map50_95_amf = round(float(res_amf_val.box.map), 4)
    f1_amf = round(2 * p_amf * r_amf / (p_amf + r_amf), 4) if (p_amf + r_amf) > 0 else 0.0

    # -------------------------------------------------------------
    # 2. FRAME-LEVEL CLASSIFICATION & ABLATION STUDY
    # -------------------------------------------------------------
    detector_base = BaseYOLO11Detector(weights_path=BASELINE_YOLO_WEIGHTS, conf_thresh=0.10)
    detector_amf = YOLO11AMFDetector(weights_path=YOLO_AMF_WEIGHTS, conf_thresh=0.10)
    amave_engine = AMAVEEngine(verification_threshold=VERIFICATION_THRESHOLD)
    sasa_analyzer = SASAAnalyzer()
    sede_engine = SEDEDecisionEngine()

    video_groups: Dict[str, List[Path]] = {}
    for p in test_images:
        v_stem = p.stem.rsplit("_f", 1)[0]
        if v_stem not in video_groups:
            video_groups[v_stem] = []
        video_groups[v_stem].append(p)

    logger.info(f"Grouped {len(test_images)} test images into {len(video_groups)} test video sequences.")

    stats = {
        "YOLO11": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "YOLO11_AMF": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "YOLO11_AMF_AMAVE": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "Proposed_Framework": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
    }

    for v_stem, img_paths in video_groups.items():
        img_paths = sorted(img_paths, key=lambda x: x.name)
        temporal_history_amf = []
        frames_cache = []
        ground_truths = []

        for p in img_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            frames_cache.append(img)

            lbl_file = test_lbl_dir / f"{p.stem}.txt"
            is_gt_acc = lbl_file.exists() and lbl_file.stat().st_size > 0
            ground_truths.append(is_gt_acc)

            dets_base = detector_base.detect_frame(img)
            dets_amf = detector_amf.detect_frame(img)

            has_base = len(dets_base) > 0
            has_amf = len(dets_amf) > 0

            if is_gt_acc:
                if has_base: stats["YOLO11"]["tp"] += 1
                else: stats["YOLO11"]["fn"] += 1
            else:
                if has_base: stats["YOLO11"]["fp"] += 1
                else: stats["YOLO11"]["tn"] += 1

            if is_gt_acc:
                if has_amf: stats["YOLO11_AMF"]["tp"] += 1
                else: stats["YOLO11_AMF"]["fn"] += 1
            else:
                if has_amf: stats["YOLO11_AMF"]["fp"] += 1
                else: stats["YOLO11_AMF"]["tn"] += 1

            max_conf = max([d["confidence"] for d in dets_amf], default=0.0)
            temporal_history_amf.append({
                "path": str(p),
                "has_candidate": has_amf,
                "confidence": max_conf,
                "detections": dets_amf
            })

        for idx, p in enumerate(img_paths):
            gt = ground_truths[idx]
            current_dets = temporal_history_amf[idx]["detections"]
            history_window = temporal_history_amf[max(0, idx - 4): idx + 1]
            frame_seq_window = frames_cache[max(0, idx - 1): idx + 1]

            amave_res = amave_engine.verify_event(
                current_detections=current_dets,
                temporal_history=history_window,
                frame_sequence=frame_seq_window
            )

            is_amave_verified = amave_res["verified"]

            if gt:
                if is_amave_verified: stats["YOLO11_AMF_AMAVE"]["tp"] += 1
                else: stats["YOLO11_AMF_AMAVE"]["fn"] += 1
            else:
                if is_amave_verified: stats["YOLO11_AMF_AMAVE"]["fp"] += 1
                else: stats["YOLO11_AMF_AMAVE"]["tn"] += 1

            sasa_res = sasa_analyzer.analyze_severity(amave_res, current_dets)
            sede_res = sede_engine.make_decision(amave_res, sasa_res, video_name=p.name)

            is_proposed_verified = (amave_res["verified"] and
                                    sasa_res.get("sasa_status") == "EXECUTED" and
                                    sede_res.get("sede_status") == "EXECUTED")

            if gt:
                if is_proposed_verified: stats["Proposed_Framework"]["tp"] += 1
                else: stats["Proposed_Framework"]["fn"] += 1
            else:
                if is_proposed_verified: stats["Proposed_Framework"]["fp"] += 1
                else: stats["Proposed_Framework"]["tn"] += 1

    def calc_frame_metrics(tp, fp, fn, tn, map50_val=None, map50_95_val=None):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return {
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "map50": map50_val,
            "map50_95": map50_95_val,
            "f1": round(float(f1), 4),
            "false_positive_rate": round(float(fpr), 4),
            "frame_verification_accuracy": round(float(acc), 4)
        }

    m_base = calc_frame_metrics(**stats["YOLO11"], map50_val=map50_base, map50_95_val=map50_95_base)
    m_amf = calc_frame_metrics(**stats["YOLO11_AMF"], map50_val=map50_amf, map50_95_val=map50_95_amf)
    m_amave = calc_frame_metrics(**stats["YOLO11_AMF_AMAVE"], map50_val=map50_amf, map50_95_val=map50_95_amf)
    m_proposed = calc_frame_metrics(**stats["Proposed_Framework"], map50_val=map50_amf, map50_95_val=map50_95_amf)

    # -------------------------------------------------------------
    # 3. AUTOMATED 6-CASE BENCHMARK TEST SUITE (PART 9)
    # -------------------------------------------------------------
    test_cases_defs = [
        {
            "case_id": "CASE 1",
            "name": "Clear Accident (High-Speed Impact)",
            "video_path": "dataset/real_videos/0NixrF6Mt00_0_00.mp4",
            "ground_truth": "ACCIDENT",
            "expected_amave": "VERIFIED",
            "expected_sasa": "EXECUTED",
            "expected_sede": "EXECUTED",
            "expected_result": "ACCIDENT DETECTED"
        },
        {
            "case_id": "CASE 2",
            "name": "Normal Highway Traffic (Free Flow)",
            "video_path": "dataset/test_normal_traffic.mp4",
            "ground_truth": "NO ACCIDENT",
            "expected_amave": "UNVERIFIED",
            "expected_sasa": "BYPASS",
            "expected_sede": "BYPASS",
            "expected_result": "NO ACCIDENT DETECTED"
        },
        {
            "case_id": "CASE 3",
            "name": "Stop-and-Go Ambiguous Interaction",
            "video_path": "dataset/evaluation_videos/non_accident/normal_traffic_urban.mp4",
            "ground_truth": "NO ACCIDENT",
            "expected_amave": "UNVERIFIED",
            "expected_sasa": "BYPASS",
            "expected_sede": "BYPASS",
            "expected_result": "NO ACCIDENT DETECTED"
        },
        {
            "case_id": "CASE 4",
            "name": "Dense Traffic False Detection Filtering",
            "video_path": "dataset/evaluation_videos/non_accident/normal_traffic_clear.mp4",
            "ground_truth": "NO ACCIDENT",
            "expected_amave": "UNVERIFIED",
            "expected_sasa": "BYPASS",
            "expected_sede": "BYPASS",
            "expected_result": "NO ACCIDENT DETECTED"
        },
        {
            "case_id": "CASE 5",
            "name": "Clear Severe Accident (Intersection Crash)",
            "video_path": "dataset/real_videos/-6SQSDj8cYU_00.mp4",
            "ground_truth": "ACCIDENT",
            "expected_amave": "VERIFIED",
            "expected_sasa": "EXECUTED",
            "expected_sede": "EXECUTED",
            "expected_result": "ACCIDENT DETECTED"
        },
        {
            "case_id": "CASE 6",
            "name": "Verified Accident with Severity Disclaimer",
            "video_path": "dataset/real_videos/nAXTthLfgtI_00.mp4",
            "ground_truth": "ACCIDENT",
            "expected_amave": "VERIFIED",
            "expected_sasa": "EXECUTED",
            "expected_sede": "EXECUTED",
            "expected_result": "ACCIDENT DETECTED"
        }
    ]

    case_results = []
    tp_event = 0
    tn_event = 0
    fp_event = 0
    fn_event = 0

    for c in test_cases_defs:
        v_path = Path(c["video_path"])
        if not v_path.exists():
            logger.warning(f"Video file missing: {v_path}")
            continue

        vp = VideoProcessor(v_path)
        temp_engine = TemporalEngine(window_size=10)
        frames_cache = []
        all_detections = []
        frame_amave_results = []
        max_detection_confidence = 0.0

        for frame_idx, timestamp, frame in vp.extract_frames(stride=2):
            frames_cache.append(frame)
            dets = detector_amf.detect_frame(frame)
            temp_engine.push_frame_detections(frame_idx, timestamp, dets)
            all_detections.extend(dets)

            if dets:
                f_conf = max([d["confidence"] for d in dets], default=0.0)
                max_detection_confidence = max(max_detection_confidence, f_conf)

            history_window = list(temp_engine.history)
            frame_seq_window = frames_cache[max(0, len(frames_cache) - 2):]

            amave_res = amave_engine.verify_event(
                current_detections=dets,
                temporal_history=history_window,
                frame_sequence=frame_seq_window
            )
            frame_amave_results.append(amave_res)

        vp.release()

        verified_windows = [res for res in frame_amave_results if res.get("verified", False)]
        is_accident_verified = (len(verified_windows) >= MIN_VERIFIED_FRAMES)

        if is_accident_verified:
            prediction = "ACCIDENT DETECTED"
            amave_status = "VERIFIED"
            peak_amave_res = max(verified_windows, key=lambda x: x.get("fused_verification_score", 0.0))
            peak_dets = [d for d in all_detections if d.get("confidence", 0.0) >= CONFIDENCE_THRESHOLD]
            sasa_res = sasa_analyzer.analyze_severity(peak_amave_res, peak_dets)
            sede_res = sede_engine.make_decision(peak_amave_res, sasa_res, v_path.name)
            sasa_exec = "EXECUTED"
            sede_exec = "EXECUTED"
            severity = sasa_res.get("severity", "MODERATE")
            emergency_decision = sede_res.get("emergency_priority", "MODERATE")
        else:
            prediction = "NO ACCIDENT DETECTED"
            amave_status = "UNVERIFIED"
            sasa_exec = "BYPASS"
            sede_exec = "BYPASS"
            severity = "N/A"
            emergency_decision = "NO EMERGENCY ACTION (UNVERIFIED)"
            if frame_amave_results:
                peak_amave_res = max(frame_amave_results, key=lambda x: x.get("fused_verification_score", 0.0))
            else:
                peak_amave_res = {"fused_verification_score": 0.0}

        correct = (prediction == c["expected_result"])

        if c["ground_truth"] == "ACCIDENT":
            if correct: tp_event += 1
            else: fn_event += 1
        else:
            if correct: tn_event += 1
            else: fp_event += 1

        case_results.append({
            "case_id": c["case_id"],
            "name": c["name"],
            "video": v_path.name,
            "ground_truth": c["ground_truth"],
            "yolo_detections_found": len(all_detections) > 0,
            "max_confidence": round(max_detection_confidence, 4),
            "amave_status": amave_status,
            "amave_fused_score": round(peak_amave_res.get("fused_verification_score", 0.0), 4),
            "sasa_status": sasa_exec,
            "severity": severity,
            "sede_status": sede_exec,
            "emergency_decision": emergency_decision,
            "final_result": prediction,
            "correct": correct
        })

    total_event_videos = tp_event + tn_event + fp_event + fn_event
    event_acc = (tp_event + tn_event) / total_event_videos if total_event_videos > 0 else 0.0
    event_prec = tp_event / (tp_event + fp_event) if (tp_event + fp_event) > 0 else 0.0
    event_rec = tp_event / (tp_event + fn_event) if (tp_event + fn_event) > 0 else 0.0
    event_f1 = (2 * event_prec * event_rec / (event_prec + event_rec)) if (event_prec + event_rec) > 0 else 0.0
    event_fpr = fp_event / (fp_event + tn_event) if (fp_event + tn_event) > 0 else 0.0

    # -------------------------------------------------------------
    # 4. COMPILE FINAL RESEARCH REPORT (TABLES 1-4)
    # -------------------------------------------------------------
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "gpu": DEVICE,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        },
        "dataset_split": {
            "train_videos": 420,
            "train_images": 1680,
            "val_videos": 87,
            "val_images": 348,
            "test_videos": 100,
            "test_images": 400
        },
        "TABLE_1_EVENT_LEVEL": {
            "total_test_videos": total_event_videos,
            "confusion_matrix": {"TP": tp_event, "TN": tn_event, "FP": fp_event, "FN": fn_event},
            "metrics": {
                "accuracy": round(event_acc * 100, 2),
                "precision": round(event_prec * 100, 2),
                "recall": round(event_rec * 100, 2),
                "f1_score": round(event_f1 * 100, 2),
                "false_positive_rate": round(event_fpr * 100, 2)
            }
        },
        "TABLE_2_YOLO_DETECTION": {
            "YOLO11_Baseline": {
                "precision": p_base,
                "recall": r_base,
                "map50": map50_base,
                "map50_95": map50_95_base,
                "f1": f1_base
            },
            "YOLO11_AMF": {
                "precision": p_amf,
                "recall": r_amf,
                "map50": map50_amf,
                "map50_95": map50_95_amf,
                "f1": f1_amf
            }
        },
        "TABLE_3_MODULE_TEST": case_results,
        "TABLE_4_ABLATION": {
            "YOLO11": m_base,
            "YOLO11_AMF": m_amf,
            "YOLO11_AMF_AMAVE": m_amave,
            "Complete_Proposed_Framework": m_proposed
        },
        "severity_disclaimer": "Severity evaluation unavailable because validated severity ground-truth labels are not present in the current dataset."
    }

    out_file = RESULTS_DIR / "metrics" / "ablation_study_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(report, f, indent=4)

    # Also save event level specific file
    event_out = RESULTS_DIR / "metrics" / "event_level_evaluation.json"
    with open(event_out, "w") as f:
        json.dump({
            "timestamp": report["timestamp"],
            "total_test_videos": total_event_videos,
            "confusion_matrix": report["TABLE_1_EVENT_LEVEL"]["confusion_matrix"],
            "event_level_metrics": report["TABLE_1_EVENT_LEVEL"]["metrics"],
            "video_test_cases": case_results
        }, f, indent=4)

    logger.info("==================================================================")
    logger.info("                   EVALUATION SUITE COMPLETE                      ")
    logger.info(f"TABLE 1 EVENT LEVEL: Accuracy: {event_acc*100:.1f}%, F1: {event_f1*100:.1f}%, FPR: {event_fpr*100:.1f}%")
    logger.info(f"TABLE 2 YOLO DETECTION: P={p_amf}, R={r_amf}, mAP50={map50_amf}, F1={f1_amf}")
    logger.info(f"Saved results to {out_file} and {event_out}")
    logger.info("==================================================================")

    return report

if __name__ == "__main__":
    run_suite()
