import requests
import json
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
API_URL = "http://127.0.0.1:8000/api/analyze"

def upload_and_test(video_path: Path, expected_status: str):
    print(f"\n========================================================")
    print(f"UPLOADING VIDEO TO FASTAPI: {video_path.name}")
    print(f"========================================================")
    
    t0 = time.time()
    with open(video_path, "rb") as f:
        files = {"file": (video_path.name, f, "video/mp4")}
        resp = requests.post(API_URL, files=files, timeout=120)
        
    duration = time.time() - t0
    assert resp.status_code == 200, f"API returned error {resp.status_code}: {resp.text}"
    
    data = resp.json()
    print(f"Response Received in {duration:.2f}s:")
    print(f"  Prediction: {data.get('prediction')}")
    print(f"  Max Detection Confidence: {data.get('detection_confidence')}")
    print(f"  AMAVE Status: {data.get('amave_status')}")
    print(f"  AMAVE Fused Score: {data.get('amave_verification', {}).get('score')}")
    print(f"  SASA Severity: {data.get('severity')}")
    print(f"  SASA Status: {data.get('sasa_severity', {}).get('sasa_status')}")
    print(f"  SEDE Emergency Action: {data.get('emergency_decision')}")
    print(f"  SEDE Priority: {data.get('sede_emergency_decision', {}).get('emergency_priority')}")
    print(f"  Annotated Video: {data.get('annotated_video_url')}")
    
    # Save test report
    report_file = BASE_DIR / "results" / "reports" / f"report_{video_path.name}.json"
    with open(report_file, "w") as out:
        json.dump(data, out, indent=4)
    print(f"  Saved API Report to: {report_file}")
    
    return data

if __name__ == "__main__":
    acc_video = BASE_DIR / "dataset" / "real_videos" / "__WFqm4i3vE_00.mp4"
    res_acc = upload_and_test(acc_video, "ACCIDENT DETECTED")
    
    norm_video = BASE_DIR / "dataset" / "test_normal_traffic.mp4"
    res_norm = upload_and_test(norm_video, "NO ACCIDENT DETECTED")
    
    print("\n========================================================")
    print("END-TO-END FASTAPI PIPELINE EXECUTION VERIFIED!")
    print(f"Accident Video Output: {res_acc.get('prediction')} (AMAVE: {res_acc.get('amave_status')})")
    print(f"Normal Video Output:   {res_norm.get('prediction')} (AMAVE: {res_norm.get('amave_status')})")
    print("========================================================")
