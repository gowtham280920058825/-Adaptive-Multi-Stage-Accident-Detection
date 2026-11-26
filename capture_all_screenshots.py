"""
Automated Screenshot Capture Engine
Captures 16 authentic, high-resolution screenshots from the running application for viva and final project report.
"""

import os
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

WORKSPACE = Path(__file__).resolve().parent
OUTPUT_DIR = WORKSPACE / "project_output_screenshots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
BASE_URL = "http://127.0.0.1:8000"

VIDEOS = {
    "case1": str(WORKSPACE / "dataset" / "evaluation_videos" / "accident" / "__WFqm4i3vE_00.mp4"),
    "case2": str(WORKSPACE / "dataset" / "evaluation_videos" / "accident" / "nAXTthLfgtI_00.mp4"),
    "normal_clear": str(WORKSPACE / "dataset" / "evaluation_videos" / "non_accident" / "normal_traffic_clear.mp4"),
    "normal_urban": str(WORKSPACE / "dataset" / "evaluation_videos" / "non_accident" / "normal_traffic_urban.mp4"),
    "failure": str(WORKSPACE / "dataset" / "real_videos" / "_9tp71S2Gwg_00.mp4")
}

def run_capture():
    print("Starting Playwright screenshot capture engine...")
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME_PATH, headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})

        # =========================================================================
        # 1. SCREENSHOT 01: Main Dashboard (Initial Ingestion State)
        # =========================================================================
        print("\n[1/16] Capturing 01_dashboard.png...")
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(1)
        page.screenshot(path=str(OUTPUT_DIR / "01_dashboard.png"))
        print("  -> Saved 01_dashboard.png")

        # =========================================================================
        # 2. SCREENSHOT 02: Video Input Selected
        # =========================================================================
        print("\n[2/16] Capturing 02_video_input.png...")
        page.set_input_files("#inputVideoFile", VIDEOS["case1"])
        time.sleep(1.5)
        page.screenshot(path=str(OUTPUT_DIR / "02_video_input.png"))
        print("  -> Saved 02_video_input.png")

        # Execute analysis on Case 1 (__WFqm4i3vE_00.mp4)
        print("Executing full analysis on Case 1 (__WFqm4i3vE_00.mp4)...")
        page.click("#btnExecuteAnalysis")
        page.wait_for_selector("#view-stage-6:not(.hidden-view)", timeout=90000)
        time.sleep(2)
        print("  -> Case 1 pipeline completed.")

        # =========================================================================
        # 3. SCREENSHOT 03: Video Preprocessing
        # =========================================================================
        print("\n[3/16] Capturing 03_video_preprocessing.png...")
        page.click("#node-stage-1")
        time.sleep(0.8)
        page.screenshot(path=str(OUTPUT_DIR / "03_video_preprocessing.png"))
        print("  -> Saved 03_video_preprocessing.png")

        # =========================================================================
        # 4. SCREENSHOT 04: YOLO11-AMF Candidate Detection
        # =========================================================================
        print("\n[4/16] Capturing 04_yolo11_amf_detection.png...")
        page.click("#node-stage-2")
        time.sleep(0.8)
        page.screenshot(path=str(OUTPUT_DIR / "04_yolo11_amf_detection.png"))
        print("  -> Saved 04_yolo11_amf_detection.png")

        # =========================================================================
        # 5. SCREENSHOT 05: Accident Candidate Proposal Stage
        # =========================================================================
        print("\n[5/16] Capturing 05_accident_candidate.png...")
        # Focus on stage 02 candidate view
        page.screenshot(path=str(OUTPUT_DIR / "05_accident_candidate.png"))
        print("  -> Saved 05_accident_candidate.png")

        # =========================================================================
        # 6. SCREENSHOT 06: AMAVE Verification Stage
        # =========================================================================
        print("\n[6/16] Capturing 06_amave_verification.png...")
        page.click("#node-stage-3")
        time.sleep(0.8)
        page.screenshot(path=str(OUTPUT_DIR / "06_amave_verification.png"))
        print("  -> Saved 06_amave_verification.png")

        # =========================================================================
        # 7. SCREENSHOT 08: SASA Severity Assessment
        # =========================================================================
        print("\n[7/16] Capturing 08_sasa_severity.png...")
        page.click("#node-stage-4")
        time.sleep(0.8)
        page.screenshot(path=str(OUTPUT_DIR / "08_sasa_severity.png"))
        print("  -> Saved 08_sasa_severity.png")

        # =========================================================================
        # 8. SCREENSHOT 09: SEDE Decision Support
        # =========================================================================
        print("\n[8/16] Capturing 09_sede_emergency_decision.png...")
        page.click("#node-stage-5")
        time.sleep(0.8)
        page.screenshot(path=str(OUTPUT_DIR / "09_sede_emergency_decision.png"))
        print("  -> Saved 09_sede_emergency_decision.png")

        # =========================================================================
        # 9. SCREENSHOT 10 & 11: End-to-End Pipeline & Accident Case 1 Result
        # =========================================================================
        print("\n[9/16] Capturing 10_end_to_end_result.png & 11_accident_case_1.png...")
        page.click("#node-stage-6")
        time.sleep(0.8)
        page.screenshot(path=str(OUTPUT_DIR / "10_end_to_end_result.png"))
        page.screenshot(path=str(OUTPUT_DIR / "11_accident_case_1.png"))
        print("  -> Saved 10_end_to_end_result.png and 11_accident_case_1.png")

        # =========================================================================
        # 10. SCREENSHOT 07 & 13: Normal Clear Traffic (AMAVE Unverified / Suppression)
        # =========================================================================
        print("\n[10/16] Running Normal Clear Traffic (normal_traffic_clear.mp4)...")
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(1)
        page.set_input_files("#inputVideoFile", VIDEOS["normal_clear"])
        time.sleep(1)
        page.click("#btnExecuteAnalysis")
        page.wait_for_selector("#view-stage-6:not(.hidden-view)", timeout=90000)
        time.sleep(2)

        # Stage 03 AMAVE Unverified View (Screenshot 07)
        page.click("#node-stage-3")
        time.sleep(0.8)
        page.screenshot(path=str(OUTPUT_DIR / "07_amave_no_accident.png"))
        print("  -> Saved 07_amave_no_accident.png")

        # Stage 06 Normal Clear Result (Screenshot 13)
        page.click("#node-stage-6")
        time.sleep(0.8)
        page.screenshot(path=str(OUTPUT_DIR / "13_normal_clear.png"))
        print("  -> Saved 13_normal_clear.png")

        # =========================================================================
        # 11. SCREENSHOT 12: Accident Case 2 (nAXTthLfgtI_00.mp4)
        # =========================================================================
        print("\n[11/16] Running Accident Case 2 (nAXTthLfgtI_00.mp4)...")
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(1)
        page.set_input_files("#inputVideoFile", VIDEOS["case2"])
        time.sleep(1)
        page.click("#btnExecuteAnalysis")
        page.wait_for_selector("#view-stage-6:not(.hidden-view)", timeout=90000)
        time.sleep(2)
        page.screenshot(path=str(OUTPUT_DIR / "12_accident_case_2.png"))
        print("  -> Saved 12_accident_case_2.png")

        # =========================================================================
        # 12. SCREENSHOT 14: Normal Urban Traffic (normal_traffic_urban.mp4)
        # =========================================================================
        print("\n[12/16] Running Normal Urban Traffic (normal_traffic_urban.mp4)...")
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(1)
        page.set_input_files("#inputVideoFile", VIDEOS["normal_urban"])
        time.sleep(1)
        page.click("#btnExecuteAnalysis")
        page.wait_for_selector("#view-stage-6:not(.hidden-view)", timeout=90000)
        time.sleep(2)
        page.screenshot(path=str(OUTPUT_DIR / "14_normal_urban.png"))
        print("  -> Saved 14_normal_urban.png")

        # =========================================================================
        # 13. SCREENSHOT 15: Failure Case Distant Accident (_9tp71S2Gwg_00.mp4)
        # =========================================================================
        print("\n[13/16] Running Failure Case Distant Accident (_9tp71S2Gwg_00.mp4)...")
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(1)
        page.set_input_files("#inputVideoFile", VIDEOS["failure"])
        time.sleep(1)
        page.click("#btnExecuteAnalysis")
        page.wait_for_selector("#view-stage-6:not(.hidden-view)", timeout=90000)
        time.sleep(2)
        page.screenshot(path=str(OUTPUT_DIR / "15_failure_case_distant_accident.png"))
        print("  -> Saved 15_failure_case_distant_accident.png")

        # =========================================================================
        # 14. SCREENSHOT 16: FastAPI Swagger UI API Documentation & Schema
        # =========================================================================
        print("\n[14/16] Capturing 16_api_output.png (FastAPI Swagger UI)...")
        page.goto(f"{BASE_URL}/docs")
        page.wait_for_load_state("networkidle")
        time.sleep(1.5)
        
        # Click on /api/analyze to expand it
        try:
            btn_analyze = page.locator("button.opblock-summary-control:has-text('/api/analyze')")
            if btn_analyze.count() > 0:
                btn_analyze.click()
                time.sleep(1)
        except Exception as e:
            print("Could not expand /api/analyze in Swagger UI:", e)

        page.screenshot(path=str(OUTPUT_DIR / "16_api_output.png"))
        print("  -> Saved 16_api_output.png")

        browser.close()

    print("\n=======================================================")
    print("ALL 16 REAL OUTPUT SCREENSHOTS SUCCESSFULLY CAPTURED!")
    print(f"Destination: {OUTPUT_DIR}")
    print("=======================================================")

if __name__ == "__main__":
    run_capture()
