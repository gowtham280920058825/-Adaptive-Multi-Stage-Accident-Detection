"""
SASA Severity Feature Extractor
Extracts quantitative, normalized [0.0, 1.0] visual scene features from detections, AMAVE V2 verification scores, and scene metadata.
"""

from typing import List, Dict, Any
import numpy as np

def extract_severity_features(
    detections: List[Dict[str, Any]],
    amave_result: Dict[str, Any] = None,
    scene_meta: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Extracts normalized [0, 1] feature vector for accident severity estimation.
    Tracks available vs unavailable features and generates plain-language explanations.
    """
    available_features = []
    unavailable_features = []
    explanations = []

    # 1. Collision Type Hazard (H_type)
    if scene_meta and "type" in scene_meta and pd_not_na(scene_meta["type"]):
        c_type = str(scene_meta["type"]).lower()
        h_type = {
            "head-on": 1.00,
            "t-bone": 0.85,
            "rear-end": 0.65,
            "sideswipe": 0.50,
            "single": 0.55
        }.get(c_type, 0.65)
        available_features.append("collision_type")
        explanations.append(f"Collision geometry: {c_type.upper()} hazard ({h_type:.2f})")
    else:
        h_type = 0.65  # Fallback default
        unavailable_features.append("collision_type")
        explanations.append("Collision geometry: Metadata unavailable (Default 0.65)")

    # 2. Rollover Flag (F_roll)
    if scene_meta and "rollover" in scene_meta and pd_not_na(scene_meta["rollover"]):
        f_roll = float(int(scene_meta["rollover"]))
        available_features.append("rollover_flag")
        if f_roll > 0:
            explanations.append("Critical Hazard: Vehicle rollover detected")
        else:
            explanations.append("Structural Stability: No vehicle rollover")
    else:
        f_roll = 0.0
        unavailable_features.append("rollover_flag")
        explanations.append("Rollover status: Metadata unavailable (Default 0.0)")

    # 3. Vehicle Count Score (N_veh)
    if detections:
        v_count = max(1, len(detections))
        available_features.append("vehicle_count")
    elif scene_meta and "involved_vehicles" in scene_meta:
        v_count = int(scene_meta["involved_vehicles"])
        available_features.append("vehicle_count")
    else:
        v_count = 1
        unavailable_features.append("vehicle_count")

    n_veh = min(1.0, float(v_count) / 4.0)
    explanations.append(f"Vehicle involvement: {v_count} detected box(es) (Normalized {n_veh:.2f})")

    # 4. Motion Evidence Score (S_motion)
    if amave_result and "motion_consistency_score" in amave_result:
        s_motion = float(amave_result["motion_consistency_score"])
        available_features.append("optical_flow_motion")
        explanations.append(f"Kinetic Impact Evidence: Optical flow motion score ({s_motion:.2f})")
    else:
        s_motion = 0.50
        unavailable_features.append("optical_flow_motion")
        explanations.append("Kinetic Motion: AMAVE Stage 3 motion unavailable (Default 0.50)")

    # 5. Scene Layout Hazard (H_layout)
    if scene_meta and "scene_layout" in scene_meta and pd_not_na(scene_meta["scene_layout"]):
        layout = str(scene_meta["scene_layout"]).lower()
        h_layout = {
            "highway": 1.00,
            "tunnel": 0.95,
            "grade_separated_intersection": 0.85,
            "signalized_intersection": 0.75,
            "city_street": 0.60,
            "simple_intersection": 0.70,
            "parking_lot": 0.40,
            "roundabout": 0.50
        }.get(layout, 0.70)
        available_features.append("scene_layout")
        explanations.append(f"Roadway context: {layout.replace('_', ' ').title()} speed hazard ({h_layout:.2f})")
    else:
        h_layout = 0.70
        unavailable_features.append("scene_layout")
        explanations.append("Roadway context: Metadata unavailable (Default 0.70)")

    return {
        "feature_values": {
            "collision_type_score": round(h_type, 4),
            "rollover_score": round(f_roll, 4),
            "vehicle_count_score": round(n_veh, 4),
            "motion_score": round(s_motion, 4),
            "scene_layout_score": round(h_layout, 4)
        },
        "raw_vehicle_count": v_count,
        "available_features": available_features,
        "unavailable_features": unavailable_features,
        "explanations": explanations
    }

def pd_not_na(val) -> bool:
    if val is None:
        return False
    try:
        import pandas as pd
        return not pd.isna(val)
    except ImportError:
        return True

