from playwright.sync_api import sync_playwright
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
OUTPUT_PATH = WORKSPACE / "project_output_screenshots" / "16_api_output.png"
VIDEO_PATH = WORKSPACE / "dataset" / "evaluation_videos" / "accident" / "__WFqm4i3vE_00.mp4"

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=CHROME_PATH, headless=True)
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    page.goto("http://127.0.0.1:8000/docs")
    page.wait_for_load_state("networkidle")
    time.sleep(1)

    # Click /api/analyze to expand
    page.locator("button.opblock-summary-control:has-text('/api/analyze')").click()
    time.sleep(0.5)

    # Click Try it out
    page.locator("button.btn.try-out__btn").click()
    time.sleep(0.5)

    # Set file input
    page.set_input_files("input[type='file']", str(VIDEO_PATH))
    time.sleep(0.5)

    # Click Execute
    page.locator("button.btn.execute.opblock-control__btn").click()
    print("Executed in Swagger UI, waiting for response...")

    # Wait for response
    page.wait_for_selector(".live-responses-table", timeout=60000)
    time.sleep(2)

    # Scroll so both the endpoint banner and the response body with 200 OK and JSON are visible
    page.evaluate("() => { const el = document.querySelector('.live-responses-table'); if(el) el.scrollIntoView({behavior: 'instant', block: 'center'}); }")
    time.sleep(1)

    page.screenshot(path=str(OUTPUT_PATH))
    print("SUCCESS: Captured 16_api_output.png with live executed Swagger UI response!")
    browser.close()
