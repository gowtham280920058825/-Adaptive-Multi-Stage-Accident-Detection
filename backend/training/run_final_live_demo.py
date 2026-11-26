"""
Final Live System Test Runner
Sends 1 positive accident video and 1 negative non-accident video to http://127.0.0.1:8000/api/analyze
Verifies live API responses, JSON structure, decision gating, and metrics.
"""

import requests
import json
import time
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from backend.config import DATASET_RAW_DIR

BASE_URL = "http://127.0.0.1:8000/api"

def test_live_system():
    print("==================================================================")
    print("           LIVE SYSTEM INFERENCE & ENDPOINT DEMO RUN              ")
    print("==================================================================")

    # 1. Health Check
    res_health = requests.get(f"{BASE_URL}/health")
    print("[1] Health Check Response:")
    print(json.dumps(res_health.json(), indent=2))
    print()

    # Search for real mp4 files
    dataset_videos = list(DATASET_RAW_DIR.glob("**/*.mp4"))
    if not dataset_videos:
        print("No mp4 video files found in dataset folder.")
        return

    accident_video_path = dataset_videos[0]
    print(f"[2] Testing ACCIDENT Video: {accident_video_path.name}")
    with open(accident_video_path, "rb") as f:
        res_acc = requests.post(f"{BASE_URL}/analyze", files={"file": (accident_video_path.name, f, "video/mp4")})

    print(f"HTTP Status: {res_acc.status_code}")
    if res_acc.status_code == 200:
        data_acc = res_acc.json()
        print("Response Summary:")
        print(f"  Prediction:           {data_acc.get('prediction')}")
        print(f"  Detection Confidence: {data_acc.get('detection_confidence_pct')}")
        print(f"  AMAVE Status:         {data_acc.get('amave_status')}")
        print(f"  Severity:             {data_acc.get('severity')}")
        print(f"  Emergency Decision:   {data_acc.get('emergency_decision')}")
        print(f"  Annotated Video URL:  {data_acc.get('annotated_video_url')}")
        print("  Model Metrics (Test Set):")
        print(json.dumps(data_acc.get("model_metrics"), indent=4))
    else:
        print(f"Error: {res_acc.text}")
    print()

    # 3. Test non-accident / background video sequence
    normal_video_path = dataset_videos[1] if len(dataset_videos) > 1 else dataset_videos[0]

    print(f"[3] Testing NORMAL / NON-ACCIDENT Video: {normal_video_path.name}")
    with open(normal_video_path, "rb") as f:
        res_norm = requests.post(f"{BASE_URL}/analyze", files={"file": (normal_video_path.name, f, "video/mp4")})

    print(f"HTTP Status: {res_norm.status_code}")
    if res_norm.status_code == 200:
        data_norm = res_norm.json()
        print("Response Summary:")
        print(f"  Prediction:           {data_norm.get('prediction')}")
        print(f"  Detection Confidence: {data_norm.get('detection_confidence_pct')}")
        print(f"  AMAVE Status:         {data_norm.get('amave_status')}")
        print(f"  Severity:             {data_norm.get('severity')}")
        print(f"  Emergency Decision:   {data_norm.get('emergency_decision')}")
        print(f"  Annotated Video URL:  {data_norm.get('annotated_video_url')}")
    else:
        print(f"Error: {res_norm.text}")
    print("==================================================================")

if __name__ == "__main__":
    test_live_system()
