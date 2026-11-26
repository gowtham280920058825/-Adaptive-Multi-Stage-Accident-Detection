"""
AMAVE Confidence Fusion Module
Fuses multi-stage evidence scores into a final accident verification probability.
"""

from typing import Dict, Any

class ConfidenceFusion:
    def __init__(
        self,
        w_det: float = 0.30,
        w_temp: float = 0.20,
        w_motion: float = 0.30,
        w_inter: float = 0.20,
        threshold: float = 0.45
    ):
        self.w_det = w_det
        self.w_temp = w_temp
        self.w_motion = w_motion
        self.w_inter = w_inter
        self.threshold = threshold

    def fuse_scores(
        self,
        det_confidence: float,
        temporal_persistence_score: float,
        motion_consistency_score: float,
        interaction_score: float,
        passed_persistence: bool = True
    ) -> Dict[str, Any]:
        """
        Stage 5: Fuses detection confidence, temporal persistence, motion consistency, and vehicle interaction.
        Returns fused score and status: VERIFIED_ACCIDENT or UNVERIFIED_EVENT.
        """
        s_det = float(det_confidence)
        s_temp = float(temporal_persistence_score)
        s_motion = float(motion_consistency_score)
        s_inter = float(interaction_score)

        fused_score = (
            (self.w_det * s_det) +
            (self.w_temp * s_temp) +
            (self.w_motion * s_motion) +
            (self.w_inter * s_inter)
        )

        # Multi-stage verification requirement:
        # Event is verified if fused score reaches decision threshold,
        # passes temporal persistence across candidate frames,
        # and satisfies physical collision evidence:
        # 1. Multi-vehicle physical contact overlap (s_inter >= 0.35, s_det >= 0.20)
        # 2. Or kinematic impact disruption / confirmed post-impact stoppage (s_motion >= 0.50, s_det >= 0.25, s_temp >= 0.40)
        # YOLO confidence alone is NEVER sufficient for verification.
        has_collision_evidence = (
            (s_inter >= 0.35 and s_det >= 0.20) or
            (s_motion >= 0.50 and s_det >= 0.25 and s_temp >= 0.40)
        )
        is_verified = (fused_score >= self.threshold) and passed_persistence and has_collision_evidence
        verification_status = "VERIFIED_ACCIDENT" if is_verified else "UNVERIFIED_EVENT"

        if is_verified:
            reason = "VERIFIED_ACCIDENT: Satisfies multi-stage temporal, motion, and interaction evidence criteria"
        elif not passed_persistence:
            reason = "UNVERIFIED_EVENT: Transient candidate detection rejected (failed temporal persistence across consecutive frames)"
        elif not has_collision_evidence:
            reason = "UNVERIFIED_EVENT: Normal vehicular flow (no vehicle-vehicle collision overlap or confirmed crash kinematics detected)"
        else:
            reason = "UNVERIFIED_EVENT: Fused verification score below decision threshold"

        return {
            "verified": is_verified,
            "verification_status": verification_status,
            "fused_verification_score": round(float(fused_score), 4),
            "detection_confidence": round(s_det, 4),
            "temporal_persistence_score": round(s_temp, 4),
            "motion_consistency_score": round(s_motion, 4),
            "interaction_score": round(s_inter, 4),
            "reason": reason
        }
