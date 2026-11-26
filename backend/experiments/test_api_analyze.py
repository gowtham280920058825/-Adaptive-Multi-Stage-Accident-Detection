import requests
import json
from pathlib import Path

video_path = Path(__file__).resolve().parent.parent.parent / "dataset" / "real_videos" / "_0RDZgCKPGo_00.mp4"
url = "http://127.0.0.1:8000/api/analyze"

print(f"Uploading {video_path.name} to {url}...")
with open(video_path, "rb") as f:
    files = {"file": (video_path.name, f, "video/mp4")}
    resp = requests.post(url, files=files)

print("Status Code:", resp.status_code)
if resp.status_code == 200:
    data = resp.json()
    print("Prediction:", data.get("prediction"))
    print("Candidate Proposals Count:", data.get("candidate_proposals_count"))
    print("Candidate Frames Count:", data.get("candidate_frames_count"))
    print("Detection Confidence Pct:", data.get("detection_confidence_pct"))
    print("AMAVE Status:", data.get("amave_status"))
    print("AMAVE Verification Score:", data.get("amave_verification", {}).get("score"))
    print("AMAVE Reason:", data.get("amave_verification", {}).get("reason"))
    print("SASA Severity:", data.get("severity"))
    print("SEDE Emergency Decision:", data.get("emergency_decision"))
    print("Annotated Video URL:", data.get("annotated_video_url"))
    
    # Check if annotated video URL is accessible
    vid_url = f"http://127.0.0.1:8000{data.get('annotated_video_url')}"
    v_resp = requests.head(vid_url)
    print(f"Annotated Video Reachability ({vid_url}): HTTP {v_resp.status_code}")
else:
    print("Error:", resp.text)
