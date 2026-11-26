"""
SASASeverityModel
Computes weighted visual scene collision severity score S_sasa and maps to LOW, MODERATE, HIGH, CRITICAL categories.
"""

from typing import Dict, Any

class SASASeverityModel:
    def __init__(
        self,
        w_type: float = 0.30,
        w_roll: float = 0.25,
        w_veh: float = 0.20,
        w_motion: float = 0.15,
        w_layout: float = 0.10,
        t_critical: float = 0.80,
        t_high: float = 0.60,
        t_moderate: float = 0.35
    ):
        self.w_type = w_type
        self.w_roll = w_roll
        self.w_veh = w_veh
        self.w_motion = w_motion
        self.w_layout = w_layout

        self.t_critical = t_critical
        self.t_high = t_high
        self.t_moderate = t_moderate

    def predict_severity(self, feature_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes composite visual scene severity index in range [0.0, 1.0].
        Maps to LOW, MODERATE, HIGH, CRITICAL using configurable thresholds.
        """
        vals = feature_data.get("feature_values", {})

        h_type = float(vals.get("collision_type_score", 0.65))
        f_roll = float(vals.get("rollover_score", 0.0))
        n_veh = float(vals.get("vehicle_count_score", 0.25))
        s_motion = float(vals.get("motion_score", 0.50))
        h_layout = float(vals.get("scene_layout_score", 0.70))

        score = (
            (self.w_type * h_type) +
            (self.w_roll * f_roll) +
            (self.w_veh * n_veh) +
            (self.w_motion * s_motion) +
            (self.w_layout * h_layout)
        )
        score = round(float(np.clip(score, 0.0, 1.0)), 4)

        if score >= self.t_critical:
            level = "CRITICAL"
        elif score >= self.t_high:
            level = "HIGH"
        elif score >= self.t_moderate:
            level = "MODERATE"
        else:
            level = "LOW"

        return {
            "sasa_status": "EXECUTED",
            "severity": level,
            "severity_score": score,
            "features": {
                "collision_type_score": h_type,
                "rollover_score": f_roll,
                "vehicle_count_score": n_veh,
                "motion_score": s_motion,
                "scene_layout_score": h_layout
            },
            "available_features": feature_data.get("available_features", []),
            "unavailable_features": feature_data.get("unavailable_features", []),
            "explanation": feature_data.get("explanations", [])
        }

import numpy as np

