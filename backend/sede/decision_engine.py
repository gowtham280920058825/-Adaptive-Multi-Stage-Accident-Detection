"""
SEDE (Smart Emergency Decision Engine)
Master module orchestrating response policy evaluation and emergency alert recommendation generation.
Executes ONLY when AMAVE V2 confirms VERIFIED_ACCIDENT and SASA is EXECUTED.
"""

from typing import Dict, Any
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from backend.sede.response_policy import SEDEResponsePolicy
from backend.sede.alert_generator import generate_emergency_alert
from backend.utils.logging_utils import get_logger

logger = get_logger("SEDE")

class SEDEDecisionEngine:
    def __init__(self, policy_engine: SEDEResponsePolicy = None):
        self.policy_engine = policy_engine if policy_engine else SEDEResponsePolicy()

    def make_decision(
        self,
        amave_result: Dict[str, Any],
        sasa_result: Dict[str, Any],
        video_name: str = "input_video.mp4"
    ) -> Dict[str, Any]:
        """
        Executes SEDE Emergency Decision Engine pipeline if AMAVE V2 verified accident and SASA executed.
        """
        amave_verified = (
            amave_result.get("verified", False) or
            amave_result.get("verification_status") == "VERIFIED_ACCIDENT"
        )
        sasa_executed = (sasa_result.get("sasa_status") == "EXECUTED")

        if not amave_verified or not sasa_executed:
            return {
                "sede_status": "NOT_EXECUTED",
                "recommended_action": "NO EMERGENCY ACTION (UNVERIFIED)",
                "emergency_priority": "N/A",
                "reason": "SEDE NOT EXECUTED: Event unverified by AMAVE or severity analysis not applicable."
            }

        policy_eval = self.policy_engine.evaluate_policy(amave_result, sasa_result)
        alert = generate_emergency_alert(amave_result, sasa_result, policy_eval, video_name)

        logger.info(
            f"SEDE Decision EXECUTED: Priority={policy_eval['emergency_priority']} "
            f"| Score={policy_eval['priority_score']:.4f} | Action={policy_eval['recommended_action']}"
        )
        return alert

