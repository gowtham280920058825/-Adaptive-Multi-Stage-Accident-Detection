import os
import sys
import json
import time
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

API_URL = "http://127.0.0.1:8000/api/analyze"
RESULTS_DIR = BASE_DIR / "results"

test_scenarios = [
    {
        "name": "Accident Video (__WFqm4i3vE_00.mp4)",
        "file": BASE_DIR / "dataset" / "evaluation_videos" / "accident" / "__WFqm4i3vE_00.mp4",
        "expected_gt": "ACCIDENT",
        "expected_prediction": "ACCIDENT DETECTED",
        "expect_sasa": True,
        "expect_sede": True
    },
    {
        "name": "Low-Light Accident Video (nAXTthLfgtI_00.mp4)",
        "file": BASE_DIR / "dataset" / "evaluation_videos" / "accident" / "nAXTthLfgtI_00.mp4",
        "expected_gt": "ACCIDENT",
        "expected_prediction": "ACCIDENT DETECTED",
        "expect_sasa": True,
        "expect_sede": True
    },
    {
        "name": "Normal Traffic Clear (normal_traffic_clear.mp4)",
        "file": BASE_DIR / "dataset" / "evaluation_videos" / "non_accident" / "normal_traffic_clear.mp4",
        "expected_gt": "NO ACCIDENT",
        "expected_prediction": "NO ACCIDENT DETECTED",
        "expect_sasa": False,
        "expect_sede": False
    },
    {
        "name": "Difficult Urban Traffic (normal_traffic_urban.mp4)",
        "file": BASE_DIR / "dataset" / "evaluation_videos" / "non_accident" / "normal_traffic_urban.mp4",
        "expected_gt": "NO ACCIDENT",
        "expected_prediction": "NO ACCIDENT DETECTED",
        "expect_sasa": False,
        "expect_sede": False
    }
]

def run_live_validation():
    print("==========================================================================")
    print("      EXECUTING END-TO-END FASTAPI PIPELINE VALIDATION (AMAVE V2)         ")
    print("==========================================================================")
    
    validation_results = []
    
    for scenario in test_scenarios:
        v_path = scenario["file"]
        print(f"\n--- Testing Scenario: {scenario['name']} ---")
        if not v_path.exists():
            print(f"ERROR: Video file not found: {v_path}")
            continue
            
        print(f"Uploading {v_path.name} to {API_URL}...")
        t0 = time.time()
        with open(v_path, "rb") as f:
            files = {"file": (v_path.name, f, "video/mp4")}
            resp = requests.post(API_URL, files=files, timeout=120)
        elapsed = time.time() - t0
        
        if resp.status_code != 200:
            print(f"FAILED: HTTP {resp.status_code}: {resp.text}")
            continue
            
        data = resp.json()
        pred = data.get("prediction")
        det_conf = data.get("detection_confidence")
        ver_count = data.get("verified_window_count", 0)
        amave_data = data.get("amave_verification", {})
        peak_score = amave_data.get("score", 0.0)
        sasa_data = data.get("sasa_severity", {})
        sede_data = data.get("sede_emergency_decision", {})
        severity = data.get("severity", "N/A")
        action = data.get("emergency_decision", "N/A")
        ann_url = data.get("annotated_video_url")

        print(f"Received response in {elapsed:.2f}s:")
        print(f"  Prediction:         {pred}")
        print(f"  Max YOLO Conf:      {det_conf}")
        print(f"  AMAVE Verified Frm: {ver_count}")
        print(f"  Peak AMAVE Score:   {peak_score}")
        print(f"  SASA Severity:      {severity} ({sasa_data.get('sasa_status')})")
        print(f"  SEDE Action:        {action}")
        print(f"  Annotated Video:    {ann_url}")
        
        # Verify Gating
        is_pred_correct = (pred == scenario["expected_prediction"])
        
        # Verify SASA gating
        sasa_gated_correct = True
        if scenario["expect_sasa"]:
            if severity == "N/A" or sasa_data.get("sasa_status") != "EXECUTED":
                sasa_gated_correct = False
        else:
            if severity != "N/A" or sasa_data.get("sasa_status") != "BYPASSED":
                sasa_gated_correct = False
                
        # Verify SEDE gating
        sede_gated_correct = True
        if scenario["expect_sede"]:
            if "UNVERIFIED" in action or sede_data.get("sede_status") != "EXECUTED":
                sede_gated_correct = False
        else:
            if "UNVERIFIED" not in action or sede_data.get("sede_status") != "BYPASSED":
                sede_gated_correct = False
                
        # Check annotated video file exists on disk
        annotated_path = RESULTS_DIR / "annotated_videos" / f"annotated_{v_path.name}"
        video_exists = annotated_path.exists() and annotated_path.stat().st_size > 0
        
        # Check JSON report exists on disk
        report_path = RESULTS_DIR / "reports" / f"report_{v_path.stem}.json"
        report_exists = report_path.exists() and report_path.stat().st_size > 0

        record = {
            "scenario": scenario["name"],
            "video": v_path.name,
            "ground_truth": scenario["expected_gt"],
            "prediction": pred,
            "correct": is_pred_correct,
            "max_yolo_conf": det_conf,
            "verified_frames": ver_count,
            "peak_amave_score": peak_score,
            "sasa_severity": severity,
            "sasa_status": sasa_data.get("sasa_status"),
            "sasa_gated_correct": sasa_gated_correct,
            "sede_emergency_action": action,
            "sede_status": sede_data.get("sede_status"),
            "sede_gated_correct": sede_gated_correct,
            "annotated_video_exists": video_exists,
            "annotated_video_size_bytes": annotated_path.stat().st_size if video_exists else 0,
            "json_report_exists": report_exists,
            "latency_seconds": round(elapsed, 2)
        }
        validation_results.append(record)
        print(f"  [STATUS]: Prediction={'PASS' if is_pred_correct else 'FAIL'} | SASA_Gating={'PASS' if sasa_gated_correct else 'FAIL'} | SEDE_Gating={'PASS' if sede_gated_correct else 'FAIL'} | Video_Saved={'PASS' if video_exists else 'FAIL'} | Report_Saved={'PASS' if report_exists else 'FAIL'}")

    out_file = RESULTS_DIR / "reports" / "final_amave_robustness_validation.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "live_validation_scenarios": validation_results
        }, f, indent=4)
        
    print(f"\n==========================================================================")
    print(f"Saved live validation report to {out_file}")
    print("==========================================================================")

if __name__ == "__main__":
    run_live_validation()
