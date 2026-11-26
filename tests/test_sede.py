"""
Unit tests for SEDE (Smart Emergency Decision Engine)
Deterministic Scenario Tests (Cases 1 - 5)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import unittest
from backend.sede.decision_engine import SEDEDecisionEngine
from backend.sede.response_policy import SEDEResponsePolicy

class TestSEDEEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SEDEDecisionEngine()

    def test_case1_unverified_event_bypasses_sede(self):
        """Case 1: UNVERIFIED_EVENT -> SEDE NOT EXECUTED, Action: NO_ACTION."""
        amave_unverified = {
            "verified": False,
            "verification_status": "UNVERIFIED_EVENT",
            "fused_verification_score": 0.35
        }
        sasa_not_exec = {
            "sasa_status": "NOT_EXECUTED",
            "severity": "N/A"
        }
        res = self.engine.make_decision(amave_unverified, sasa_not_exec)
        self.assertEqual(res["sede_status"], "NOT_EXECUTED")
        self.assertEqual(res["recommended_action"], "NO EMERGENCY ACTION (UNVERIFIED)")
        self.assertEqual(res["emergency_priority"], "N/A")

    def test_case2_verified_low_severity(self):
        """Case 2: VERIFIED + LOW -> Priority LOW / Record Event & Monitor."""
        amave_verified = {
            "verified": True,
            "verification_status": "VERIFIED_ACCIDENT",
            "fused_verification_score": 0.50
        }
        sasa_low = {
            "sasa_status": "EXECUTED",
            "severity": "LOW",
            "severity_score": 0.20
        }
        res = self.engine.make_decision(amave_verified, sasa_low)
        self.assertEqual(res["sede_status"], "EXECUTED")
        self.assertEqual(res["emergency_priority"], "LOW")
        self.assertIn("Incident Logging", res["recommended_action"])

    def test_case3_verified_moderate_severity(self):
        """Case 3: VERIFIED + MODERATE -> Priority MODERATE / Control Room Review."""
        amave_verified = {
            "verified": True,
            "verification_status": "VERIFIED_ACCIDENT",
            "fused_verification_score": 0.58
        }
        sasa_mod = {
            "sasa_status": "EXECUTED",
            "severity": "MODERATE",
            "severity_score": 0.48
        }
        res = self.engine.make_decision(amave_verified, sasa_mod)
        self.assertEqual(res["sede_status"], "EXECUTED")
        self.assertEqual(res["emergency_priority"], "MODERATE")
        self.assertIn("Control Room Review", res["recommended_action"])

    def test_case4_verified_high_severity(self):
        """Case 4: VERIFIED + HIGH -> Priority HIGH / Emergency Medical & Police."""
        amave_verified = {
            "verified": True,
            "verification_status": "VERIFIED_ACCIDENT",
            "fused_verification_score": 0.65
        }
        sasa_high = {
            "sasa_status": "EXECUTED",
            "severity": "HIGH",
            "severity_score": 0.68
        }
        res = self.engine.make_decision(amave_verified, sasa_high)
        self.assertEqual(res["sede_status"], "EXECUTED")
        self.assertEqual(res["emergency_priority"], "HIGH")
        self.assertIn("HIGH-PRIORITY EMERGENCY RECOMMENDATION", res["recommended_action"])

    def test_case5_verified_critical_severity(self):
        """Case 5: VERIFIED + CRITICAL -> Priority CRITICAL / Immediate Ambulance & Police Advisory."""
        amave_verified = {
            "verified": True,
            "verification_status": "VERIFIED_ACCIDENT",
            "fused_verification_score": 0.85
        }
        sasa_crit = {
            "sasa_status": "EXECUTED",
            "severity": "CRITICAL",
            "severity_score": 0.88
        }
        res = self.engine.make_decision(amave_verified, sasa_crit)
        self.assertEqual(res["sede_status"], "EXECUTED")
        self.assertEqual(res["emergency_priority"], "CRITICAL")
        self.assertIn("CRITICAL EMERGENCY RESPONSE RECOMMENDATION", res["recommended_action"])

if __name__ == "__main__":
    unittest.main()
