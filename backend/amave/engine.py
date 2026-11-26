"""
AMAVE (Adaptive Multi-Stage Accident Verification Engine)
Master Module orchestrating 5-stage verification:
Stage 1: Detection confidence
Stage 2: Temporal persistence & kinematics
Stage 3: Motion consistency (Farneback Optical Flow with local background ring estimation)
Stage 4: Vehicle interaction / collision evidence
Stage 5: Confidence fusion
"""

import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from backend.config import VERIFICATION_THRESHOLD, MIN_PERSISTENCE_FRAMES, AMAVE_WEIGHTS
from backend.amave.temporal_verifier import TemporalVerifier
from backend.amave.confidence_fusion import ConfidenceFusion
from backend.utils.logging_utils import get_logger

logger = get_logger("AMAVE")

class AMAVEEngine:
    def __init__(
        self,
        verification_threshold: float = VERIFICATION_THRESHOLD,
        min_persistence_frames: int = MIN_PERSISTENCE_FRAMES,
        weights: Dict[str, float] = None
    ):
        w = weights or AMAVE_WEIGHTS
        self.verifier = TemporalVerifier(min_persistence_frames=min_persistence_frames)
        self.fusion = ConfidenceFusion(
            w_det=w.get("det", 0.30),
            w_temp=w.get("temp", 0.20),
            w_motion=w.get("motion", 0.30),
            w_inter=w.get("inter", 0.20),
            threshold=verification_threshold
        )

    def verify_event(
        self,
        current_detections: List[Dict[str, Any]],
        temporal_history: List[Dict[str, Any]],
        frame_sequence: List[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Executes genuine 5-stage AMAVE verification pipeline.
        Returns exact required output dictionary fields.
        """
        if not current_detections:
            return {
                "verified": False,
                "verification_status": "UNVERIFIED_EVENT",
                "fused_verification_score": 0.0,
                "verification_score": 0.0,
                "detection_confidence": 0.0,
                "temporal_persistence_score": 0.0,
                "motion_consistency_score": 0.0,
                "interaction_score": 0.0,
                "reason": "UNVERIFIED_EVENT: No accident candidates detected"
            }

        img_shape = (1080, 1920)
        if frame_sequence and len(frame_sequence) > 0:
            img_shape = frame_sequence[-1].shape[:2]

        # Stage 1: Detection Confidence (excluding ego-vehicle mounting borders)
        valid_dets = [
            d for d in current_detections
            if not self.verifier.is_ego_border_box(d.get("box", [0, 0, 0, 0]), img_shape[1], img_shape[0])
        ]
        if not valid_dets:
            return {
                "verified": False,
                "verification_status": "UNVERIFIED_EVENT",
                "fused_verification_score": 0.20,
                "verification_score": 0.20,
                "detection_confidence": round(float(max([d.get("confidence", 0.0) for d in current_detections], default=0.0)), 4),
                "temporal_persistence_score": 0.0,
                "motion_consistency_score": 0.20,
                "interaction_score": 0.15,
                "reason": "UNVERIFIED_EVENT: Ego-vehicle camera border artifact filtered"
            }

        det_confidence = float(max([d.get("confidence", 0.0) for d in valid_dets], default=0.0))

        # Stage 2: Temporal Persistence & Kinematic Trajectory Analysis
        temp_eval = self.verifier.evaluate_temporal_persistence(temporal_history, img_shape=img_shape)
        s_pers = float(temp_eval.get("persistence_ratio", 0.0))
        passed_persistence = bool(temp_eval.get("passed", False))

        # Stage 3: Motion Consistency (Farneback Optical Flow with local background ring)
        target_boxes = [d.get("box", []) for d in valid_dets if "box" in d]
        if frame_sequence and len(frame_sequence) >= 2:
            s_motion = self.verifier.evaluate_motion_consistency(
                frame_sequence=frame_sequence,
                target_boxes=target_boxes,
                temporal_kinematics=temp_eval,
                max_detection_conf=det_confidence
            )
        else:
            s_motion = 0.65 if passed_persistence else 0.40

        # Stage 4: Vehicle Interaction / Collision Evidence
        s_inter = self.verifier.evaluate_vehicle_interaction(valid_dets, img_shape=img_shape)

        # Stage 5: Confidence Fusion
        fusion_res = self.fusion.fuse_scores(
            det_confidence=det_confidence,
            temporal_persistence_score=s_pers,
            motion_consistency_score=s_motion,
            interaction_score=s_inter,
            passed_persistence=passed_persistence
        )

        result = {
            "verified": fusion_res["verified"],
            "verification_status": fusion_res["verification_status"],
            "fused_verification_score": fusion_res["fused_verification_score"],
            "verification_score": fusion_res["fused_verification_score"],
            "detection_confidence": fusion_res["detection_confidence"],
            "temporal_persistence_score": fusion_res["temporal_persistence_score"],
            "motion_consistency_score": fusion_res["motion_consistency_score"],
            "interaction_score": fusion_res["interaction_score"],
            "reason": fusion_res["reason"]
        }

        logger.info(
            f"AMAVE Verification: Status={result['verification_status']} | "
            f"Fused={result['fused_verification_score']:.4f} (Det={det_confidence:.2f}, Pers={s_pers:.2f}, Mot={s_motion:.2f}, Int={s_inter:.2f})"
        )
        return result
