import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import unittest
from backend.sasa.analyzer import SASAAnalyzer
from backend.sasa.severity_features import extract_severity_features
from backend.sasa.severity_model import SASASeverityModel

class TestSASAEngine(unittest.TestCase):
    def setUp(self):
        self.analyzer = SASAAnalyzer()

    def test_sasa_bypassed_when_unverified(self):
        """Test that SASA returns sasa_status NOT_EXECUTED when AMAVE returns UNVERIFIED_EVENT."""
        amave_unverified = {
            "verified": False,
            "verification_status": "UNVERIFIED_EVENT",
            "fused_verification_score": 0.35
        }
        res = self.analyzer.analyze_severity(
            amave_result=amave_unverified,
            detections=[{"box": [10, 10, 50, 50], "confidence": 0.40}]
        )
        self.assertEqual(res["sasa_status"], "NOT_EXECUTED")
        self.assertEqual(res["severity"], "N/A")

    def test_sasa_executed_when_verified(self):
        """Test that SASA executes when AMAVE returns VERIFIED_ACCIDENT."""
        amave_verified = {
            "verified": True,
            "verification_status": "VERIFIED_ACCIDENT",
            "fused_verification_score": 0.65,
            "motion_consistency_score": 0.70
        }
        scene_meta = {
            "type": "t-bone",
            "rollover": 1,
            "scene_layout": "highway"
        }
        res = self.analyzer.analyze_severity(
            amave_result=amave_verified,
            detections=[{"box": [10, 10, 50, 50], "confidence": 0.85}],
            scene_meta=scene_meta
        )
        self.assertEqual(res["sasa_status"], "EXECUTED")
        self.assertIn(res["severity"], ["LOW", "MODERATE", "HIGH", "CRITICAL"])
        self.assertGreaterEqual(res["severity_score"], 0.0)
        self.assertLessEqual(res["severity_score"], 1.0)
        self.assertIn("rollover_score", res["features"])
        self.assertIn("collision_type", res["available_features"])

    def test_missing_feature_fallback(self):
        """Test feature extraction fallback when scene metadata is missing."""
        feats = extract_severity_features(detections=[], scene_meta=None)
        self.assertIn("collision_type", feats["unavailable_features"])
        self.assertIn("rollover_flag", feats["unavailable_features"])
        self.assertEqual(feats["feature_values"]["rollover_score"], 0.0)

    def test_boundary_thresholds(self):
        """Test SASA threshold mapping logic."""
        model = SASASeverityModel(t_critical=0.80, t_high=0.60, t_moderate=0.35)
        
        # Test low feature values -> LOW
        low_data = {"feature_values": {"collision_type_score": 0.5, "rollover_score": 0.0, "vehicle_count_score": 0.25, "motion_score": 0.1, "scene_layout_score": 0.4}}
        res_low = model.predict_severity(low_data)
        self.assertEqual(res_low["severity"], "LOW")

        # Test high feature values -> CRITICAL
        crit_data = {"feature_values": {"collision_type_score": 1.0, "rollover_score": 1.0, "vehicle_count_score": 1.0, "motion_score": 0.9, "scene_layout_score": 1.0}}
        res_crit = model.predict_severity(crit_data)
        self.assertEqual(res_crit["severity"], "CRITICAL")

if __name__ == "__main__":
    unittest.main()
