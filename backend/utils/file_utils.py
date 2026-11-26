"""
File Utility Helpers
Handles file validation, path traversal security, JSON report saving, and directory management.
"""

import json
from pathlib import Path
from typing import Dict, Any

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

def validate_video_file(file_path: Path) -> bool:
    """Validate file extension and existence."""
    if not file_path.exists():
        return False
    return file_path.suffix.lower() in ALLOWED_EXTENSIONS

def save_json_report(data: Dict[str, Any], output_path: Path) -> None:
    """Save dictionary analysis report as formatted JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)

def load_json_report(input_path: Path) -> Dict[str, Any]:
    """Load JSON report."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)
