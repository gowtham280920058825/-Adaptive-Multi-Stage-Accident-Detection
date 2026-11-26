"""
Backend Health & Route Integration Tests
"""

from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from backend.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ONLINE"
    assert "gpu" in data
    assert "modules" in data

def test_model_info_endpoint():
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["active_model"] == "YOLO11-AMF"
