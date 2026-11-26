"""
SASA (Smart Accident Severity Analyzer)
Master module executing severity analysis ONLY when AMAVE V2 verifies a valid accident.
"""

from typing import List, Dict, Any
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from backend.sasa.severity_features import extract_severity_features
from backend.sasa.severity_model import SASASeverityModel
from backend.utils.logging_utils import get_logger

logger = get_logger("SASA")

class SASAAnalyzer:
    def __init__(self, model: SASASeverityModel = None):
        self.model = model if model else SASASeverityModel()

    def analyze_severity(
        self,
        amave_result: Dict[str, Any],
        detections: List[Dict[str, Any]],
        scene_meta: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Executes SASA analysis pipeline if AMAVE V2 verified accident event.
        Returns sasa_status: EXECUTED vs NOT_EXECUTED.
        """
        is_verified = (
            amave_result.get("verified", False) or
            amave_result.get("verification_status") == "VERIFIED_ACCIDENT"
        )

        if not is_verified:
            return {
                "sasa_status": "NOT_EXECUTED",
                "severity": "N/A",
                "severity_score": 0.0,
                "reason": "SASA NOT EXECUTED: Event unverified by AMAVE temporal engine."
            }

        features = extract_severity_features(
            detections=detections,
            amave_result=amave_result,
            scene_meta=scene_meta
        )
        result = self.model.predict_severity(features)

        logger.info(
            f"SASA Analysis EXECUTED: Severity={result['severity']} | Score={result['severity_score']:.4f} "
            f"| Features Used={len(result['available_features'])}"
        )
        return result

