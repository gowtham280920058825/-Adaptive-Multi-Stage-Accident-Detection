import os
import sys
import json
import time
import cv2
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from backend.config import RESULTS_DIR, BASE_DIR
DATASET_DIR = BASE_DIR / "dataset"
from backend.detection.video_processor import VideoProcessor
from backend.utils.logging_utils import get_logger

logger = get_logger("EventLevelEvaluator")

def prepare_benchmark_videos():
    eval_dir = DATASET_DIR / "evaluation_videos"
    acc_dir = eval_dir / "accident"
    norm_dir = eval_dir / "non_accident"
    acc_dir.mkdir(parents=True, exist_ok=True)
    norm_dir.mkdir(parents=True, exist_ok=True)

    # 1. Existing normal traffic
    test_normal = DATASET_DIR / "test_normal_traffic.mp4"
    if test_normal.exists():
        import shutil
        target = norm_dir / "normal_traffic_01.mp4"
        if not target.exists():
            shutil.copy(str(test_normal), str(target))

    # 2. Extract verified normal traffic clips from pre-accident segments
    normal_sources = [
        {"src": "dataset/synthetic_videos/videos/sideswipe/Town05_sideswipe_clear_00.mp4", "out": "normal_traffic_clear.mp4", "max_frames": 60},
        {"src": "dataset/real_videos/Z4kg2Ev3vhk_00.mp4", "out": "normal_traffic_urban.mp4", "max_frames": 60}
    ]

    for item in normal_sources:
        out_p = norm_dir / item["out"]
        if not out_p.exists() and Path(item["src"]).exists():
            cap = cv2.VideoCapture(item["src"])
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(out_p), fourcc, fps, (w, h))
            count = 0
            while cap.isOpened() and count < item["max_frames"]:
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
                count += 1
            cap.release()
            writer.release()
            logger.info(f"Created normal traffic clip: {out_p.name} ({count} frames)")

    # 3. Add strictly held-out accident test videos (zero training leakage)
    acc_sources = [
        "__WFqm4i3vE_00.mp4",
        "-6SQSDj8cYU_00.mp4",
        "fzWY0vLAXzI_00.mp4",
        "Z4kg2Ev3vhk_00.mp4",
        "0isplBZQftg_00.mp4"
    ]
    for s in acc_sources:
        src_p = DATASET_DIR / "real_videos" / s
        if src_p.exists():
            import shutil
            tgt_p = acc_dir / s
            if not tgt_p.exists():
                shutil.copy(str(src_p), str(tgt_p))

    return acc_dir, norm_dir

from backend.detection.yolov11_amf import YOLO11AMFDetector
from backend.amave.engine import AMAVEEngine
from backend.detection.temporal_engine import TemporalEngine
from backend.sasa.analyzer import SASAAnalyzer
from backend.sede.decision_engine import SEDEDecisionEngine
from backend.config import CONFIDENCE_THRESHOLD, MIN_VERIFIED_FRAMES, VERIFICATION_THRESHOLD, YOLO_AMF_WEIGHTS

def run_event_level_evaluation():
    logger.info("==================================================================")
    logger.info("       STARTING EVENT-LEVEL VIDEO ACCIDENT DETECTION BENCHMARK    ")
    logger.info("==================================================================")

    acc_dir, norm_dir = prepare_benchmark_videos()
    detector = YOLO11AMFDetector(weights_path=YOLO_AMF_WEIGHTS, conf_thresh=CONFIDENCE_THRESHOLD)
    amave_engine = AMAVEEngine(verification_threshold=VERIFICATION_THRESHOLD)
    sasa_analyzer = SASAAnalyzer()
    sede_engine = SEDEDecisionEngine()

    accident_videos = sorted(list(acc_dir.glob("*.mp4")))
    normal_videos = sorted(list(norm_dir.glob("*.mp4")))

    logger.info(f"Found {len(accident_videos)} accident test videos.")
    logger.info(f"Found {len(normal_videos)} normal/non-accident test videos.")

    results_table = []
    tp = 0
    fn = 0
    fp = 0
    tn = 0

    def evaluate_single_video(v_path: Path, expected_gt: str):
        vp = VideoProcessor(v_path)
        temp_engine = TemporalEngine(window_size=10)
        frames_cache = []
        all_detections = []
        frame_amave_results = []
        max_detection_confidence = 0.0

        for frame_idx, timestamp, frame in vp.extract_frames(stride=2):
            frames_cache.append(frame)
            dets = detector.detect_frame(frame)
            temp_engine.push_frame_detections(frame_idx, timestamp, dets)
            all_detections.extend(dets)

            if dets:
                frame_max_conf = max([d["confidence"] for d in dets], default=0.0)
                max_detection_confidence = max(max_detection_confidence, frame_max_conf)

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
            severity = sasa_res.get("severity", "MODERATE")
            emergency_decision = sede_res.get("emergency_priority", "MODERATE")
        else:
            prediction = "NO ACCIDENT DETECTED"
            amave_status = "UNVERIFIED"
            severity = "N/A"
            emergency_decision = "NO EMERGENCY ACTION (UNVERIFIED)"
            if frame_amave_results:
                peak_amave_res = max(frame_amave_results, key=lambda x: x.get("fused_verification_score", 0.0))
            else:
                peak_amave_res = {"fused_verification_score": 0.0}

        yolo_det = "DETECTIONS FOUND" if len(all_detections) > 0 else "NO DETECTIONS"
        correct = (prediction == ("ACCIDENT DETECTED" if expected_gt == "ACCIDENT" else "NO ACCIDENT DETECTED"))

        return {
            "test_id": f"TEST_{expected_gt[:3]}_{v_path.stem[:12]}",
            "video_name": v_path.name,
            "ground_truth": expected_gt,
            "yolo_result": yolo_det,
            "detection_confidence": round(max_detection_confidence, 4),
            "amave": amave_status,
            "amave_score": round(peak_amave_res.get("fused_verification_score", 0.0), 4),
            "sasa": severity,
            "sede": emergency_decision,
            "final_result": prediction,
            "correct": correct
        }

    # Test accident videos (Ground Truth = ACCIDENT)
    for v_path in accident_videos:
        logger.info(f"Processing Accident Video: {v_path.name}")
        entry = evaluate_single_video(v_path, expected_gt="ACCIDENT")
        if entry["correct"]:
            tp += 1
        else:
            fn += 1
        results_table.append(entry)

    # Test normal traffic videos (Ground Truth = NO ACCIDENT)
    for v_path in normal_videos:
        logger.info(f"Processing Non-Accident Video: {v_path.name}")
        entry = evaluate_single_video(v_path, expected_gt="NO ACCIDENT")
        if entry["correct"]:
            tn += 1
        else:
            fp += 1
        results_table.append(entry)

    total_videos = tp + tn + fp + fn
    accuracy = (tp + tn) / total_videos if total_videos > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    eval_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_videos": total_videos,
        "confusion_matrix": {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn
        },
        "event_level_metrics": {
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "false_positive_rate": round(fpr * 100, 2)
        },
        "video_test_cases": results_table
    }

    out_file = RESULTS_DIR / "metrics" / "event_level_evaluation.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(eval_data, f, indent=4)

    logger.info("==================================================================")
    logger.info(f"EVENT-LEVEL RESULTS: Accuracy: {eval_data['event_level_metrics']['accuracy']}% | F1: {eval_data['event_level_metrics']['f1_score']}% | FPR: {eval_data['event_level_metrics']['false_positive_rate']}%")
    logger.info(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    logger.info(f"Saved event-level evaluation to {out_file}")
    logger.info("==================================================================")

    return eval_data

if __name__ == "__main__":
    run_event_level_evaluation()
