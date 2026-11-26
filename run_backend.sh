#!/bin/bash
echo "Starting FastAPI Backend Server..."
python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
