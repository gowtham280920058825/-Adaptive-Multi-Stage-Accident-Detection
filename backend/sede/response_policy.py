"""
SEDE Response Policy Engine
Maps AMAVE V2 verification score and SASA visual severity score into deterministic emergency response priority recommendations.
"""

from typing import Dict, Any
import numpy as np

class SEDEResponsePolicy:
    def __init__(self, w_sasa: float = 0.60, w_amave: float = 0.40):
        self.w_sasa = w_sasa
        self.w_amave = w_amave

    def evaluate_policy(
        self,
        amave_result: Dict[str, Any],
        sasa_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculates Emergency Priority Score P_sede and maps to emergency response policy recommendations.
        """
        sasa_level = sasa_result.get("severity", "LOW")
        sasa_score = float(sasa_result.get("severity_score", 0.0))
        amave_score = float(amave_result.get("fused_verification_score", 0.50))

        priority_score = (self.w_sasa * sasa_score) + (self.w_amave * amave_score)
        priority_score = round(float(np.clip(priority_score, 0.0, 1.0)), 4)

        if sasa_level == "CRITICAL" or priority_score >= 0.70:
            priority = "CRITICAL"
            action = "CRITICAL EMERGENCY RESPONSE RECOMMENDATION: Immediate Ambulance & Police Dispatch Advisory"
            desc = "High-impact collision evidence detected. Immediate emergency medical & traffic police dispatch advisory generated for control room operators."
            steps = [
                "Issue Priority Level 1 Control Room Alert",
                "Recommend Immediate Emergency Medical Services (Ambulance) Dispatch",
                "Recommend Traffic Police & Roadside Clearance Unit Dispatch",
                "Trigger Upstream Variable Message Sign (VMS) Warning Advisory"
            ]
        elif sasa_level in ["SEVERE", "HIGH"] or priority_score >= 0.55:
            priority = "HIGH"
            action = "HIGH-PRIORITY EMERGENCY RECOMMENDATION: Traffic Control & Emergency Medical Notification"
            desc = "Moderate-to-high severity accident confirmed. Priority control room notification and emergency medical response recommended."
            steps = [
                "High-Priority Control Room Operator Alert",
                "Recommend Emergency Medical Response Notification",
                "Recommend Police Assistance Advisory"
            ]
        elif sasa_level == "MODERATE" or priority_score >= 0.35:
            priority = "MODERATE"
            action = "MODERATE RESPONSE RECOMMENDATION: Traffic Control Room Review Advisory"
            desc = "Verified traffic accident event with moderate severity. Control room operator review advisory queued."
            steps = [
                "Queue Event for Operator Review",
                "Notify Area Traffic Monitoring Operator",
                "Maintain Continuous Camera Surveillance"
            ]
        else:
            priority = "LOW"
            action = "LOW PRIORITY RECOMMENDATION: Incident Logging & Continuous Monitoring"
            desc = "Verified minor incident. Logged for traffic analysis with no immediate emergency dispatch escalation."
            steps = [
                "Log Event to Incident Database",
                "Continue Standard Video Monitoring"
            ]

        rationale = [
            f"AMAVE Verified (Score: {amave_score:.4f})",
            f"SASA Visual Severity: {sasa_level} (Score: {sasa_score:.4f})",
            f"Computed Priority Score: {priority_score:.4f}"
        ]

        return {
            "sede_status": "EXECUTED",
            "emergency_priority": priority,
            "priority_score": priority_score,
            "recommended_action": action,
            "description": desc,
            "action_steps": steps,
            "evidence_rationale": rationale
        }

