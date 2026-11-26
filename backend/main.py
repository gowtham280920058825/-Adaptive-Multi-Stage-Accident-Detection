"""
FastAPI Server Entry Point
Adaptive Multi-Stage Framework for Vision-Based Road Accident Detection and Intelligent Emergency Alert System
"""

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from backend.config import BASE_DIR, RESULTS_DIR
from backend.api.routes import router as api_router
from backend.utils.logging_utils import get_logger

from contextlib import asynccontextmanager

logger = get_logger("MainServer")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("==========================================================================")
    logger.info(" ADAPTIVE MULTI-STAGE ROAD ACCIDENT DETECTION SYSTEM INITIALIZED ")
    logger.info(" Baseline: YOLO11 | Base Paper: YOLO11-AMF ")
    logger.info(" Proposed Research Modules: AMAVE -> SASA -> SEDE ")
    logger.info(" Server Running at: http://127.0.0.1:8000 ")
    logger.info("==========================================================================")
    yield

app = FastAPI(
    title="Adaptive Multi-Stage Framework for Road Accident Detection",
    description="M.Sc. R&D Prototype featuring YOLO11-AMF, AMAVE, SASA, and SEDE.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local customer dashboard running
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Router
app.include_router(api_router, prefix="/api")

if RESULTS_DIR.exists():
    app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
