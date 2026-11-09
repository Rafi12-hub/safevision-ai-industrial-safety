"""
Comprehensive Test Suite for AI-Based Mechanical Lab & Industrial Worker Safety Monitoring System.
Validates all required scenarios demanded by the specification:
Test 1: Worker wearing safety helmet and safety shoes -> GREEN - SAFE
Test 2: Worker wearing helmet but missing safety shoes -> RED - NOT SAFE
Test 3: Worker wearing safety shoes but missing safety helmet -> RED - NOT SAFE
Test 4: Worker wearing neither safety helmet nor safety shoes -> RED - NOT SAFE
Test 5: Worker's feet outside bottom camera frame -> YELLOW - UNKNOWN
Test 6: Multi-worker simultaneous verification -> Individual results per worker ID
Test 7: Spatial association verification -> Helmet/shoes elsewhere in frame not assigned
Test 8: Model unavailable -> Transparent diagnostic message, never fabricates detections
"""

import pytest
import os
import sys
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from config import SafetyStatus, STATUS_COLORS
from detection.safety_checker import SafetyVerifier
from detection.tracker import WorkerTracker
from detection.detector import SafetyDetector
from database.safety_records import add_safety_record, get_all_records_df


@pytest.fixture
def verifier():
    return SafetyVerifier(frame_edge_margin=15, hysteresis_window=5, allow_safety_cap=True)


@pytest.fixture
def tracker():
    return WorkerTracker(iou_threshold=0.35, cooldown_sec=5.0)


def test_scenario_1_safe_helmet_and_shoes(verifier):
    """Test 1: Worker wearing both safety helmet and safety shoes -> GREEN - SAFE."""
    person_box = [200.0, 50.0, 360.0, 420.0]
    helmet_box = {"bbox": [250.0, 40.0, 310.0, 85.0], "confidence": 0.94, "class_id": 1, "name": "safety_helmet"}
    shoe_box = {"bbox": [220.0, 390.0, 340.0, 420.0], "confidence": 0.89, "class_id": 3, "name": "safety_shoes"}

    res = None
    for _ in range(5):
        res = verifier.evaluate_worker(
            worker_id="Worker 001",
            person_box=person_box,
            person_confidence=0.95,
            helmet_boxes=[helmet_box],
            cap_boxes=[],
            shoe_boxes=[shoe_box],
            frame_width=640,
            frame_height=480
        )

    assert res["status"] == SafetyStatus.SAFE
    assert res["helmet_detected"] is True
    assert res["shoes_detected"] is True
    assert res["color"] == STATUS_COLORS[SafetyStatus.SAFE]
    assert "SAFE" in res["details"]


def test_scenario_2_missing_safety_shoes(verifier):
    """Test 2: Worker wearing helmet but missing safety shoes -> RED - NOT SAFE."""
    person_box = [200.0, 50.0, 360.0, 420.0]
    helmet_box = {"bbox": [250.0, 40.0, 310.0, 85.0], "confidence": 0.92, "class_id": 1, "name": "safety_helmet"}

    res = None
    for _ in range(5):
        res = verifier.evaluate_worker(
            worker_id="Worker 002",
            person_box=person_box,
            person_confidence=0.94,
            helmet_boxes=[helmet_box],
            cap_boxes=[],
            shoe_boxes=[],
            frame_width=640,
            frame_height=480
        )

    assert res["status"] == SafetyStatus.NOT_SAFE
    assert res["helmet_detected"] is True
    assert res["shoes_detected"] is False
    assert res["color"] == STATUS_COLORS[SafetyStatus.NOT_SAFE]
    assert "Missing Safety Shoes" in res["details"]


def test_scenario_3_missing_safety_helmet(verifier):
    """Test 3: Worker wearing safety shoes but missing helmet -> RED - NOT SAFE."""
    person_box = [200.0, 50.0, 360.0, 420.0]
    shoe_box = {"bbox": [220.0, 390.0, 340.0, 420.0], "confidence": 0.90, "class_id": 3, "name": "safety_shoes"}

    res = None
    for _ in range(5):
        res = verifier.evaluate_worker(
            worker_id="Worker 003",
            person_box=person_box,
            person_confidence=0.93,
            helmet_boxes=[],
            cap_boxes=[],
            shoe_boxes=[shoe_box],
            frame_width=640,
            frame_height=480
        )

    assert res["status"] == SafetyStatus.NOT_SAFE
    assert res["helmet_detected"] is False
    assert res["shoes_detected"] is True
    assert res["color"] == STATUS_COLORS[SafetyStatus.NOT_SAFE]
    assert "Missing Safety Helmet" in res["details"]


def test_scenario_4_missing_both_ppe(verifier):
    """Test 4: Worker wearing neither helmet nor shoes -> RED - NOT SAFE."""
    person_box = [200.0, 50.0, 360.0, 420.0]

    res = None
    for _ in range(5):
        res = verifier.evaluate_worker(
            worker_id="Worker 004",
            person_box=person_box,
            person_confidence=0.92,
            helmet_boxes=[],
            cap_boxes=[],
            shoe_boxes=[],
            frame_width=640,
            frame_height=480
        )

    assert res["status"] == SafetyStatus.NOT_SAFE
    assert res["helmet_detected"] is False
    assert res["shoes_detected"] is False
    assert res["color"] == STATUS_COLORS[SafetyStatus.NOT_SAFE]
    assert "Missing Safety Helmet, Safety Shoes" in res["details"]


def test_scenario_5_feet_outside_camera_frame(verifier):
    """Test 5: Worker's feet outside bottom of camera frame -> YELLOW - UNKNOWN."""
    # Worker extends beyond bottom edge (frame_height=480, margin=15 => y2 >= 465)
    person_box = [200.0, 60.0, 360.0, 475.0]  # py2=475 > 465
    helmet_box = {"bbox": [250.0, 50.0, 310.0, 95.0], "confidence": 0.91, "class_id": 1, "name": "safety_helmet"}

    res = None
    for _ in range(5):
        res = verifier.evaluate_worker(
            worker_id="Worker 005",
            person_box=person_box,
            person_confidence=0.95,
            helmet_boxes=[helmet_box],
            cap_boxes=[],
            shoe_boxes=[],
            frame_width=640,
            frame_height=480
        )

    assert res["feet_visible"] is False
    assert res["status"] == SafetyStatus.UNKNOWN
    assert res["color"] == STATUS_COLORS[SafetyStatus.UNKNOWN]
    assert "UNKNOWN" in res["details"]


def test_scenario_6_multi_worker_tracking(verifier, tracker):
    """Test 6: Multiple workers entering simultaneously -> Individual tracking IDs & independent PPE checks."""
    # Worker A on left (full PPE)
    person_a = {"bbox": [50.0, 60.0, 180.0, 420.0], "confidence": 0.94}
    helmet_a = {"bbox": [90.0, 50.0, 140.0, 90.0], "confidence": 0.91, "class_id": 1}
    shoe_a = {"bbox": [80.0, 390.0, 150.0, 420.0], "confidence": 0.88, "class_id": 3}

    # Worker B on right (missing shoes)
    person_b = {"bbox": [380.0, 60.0, 510.0, 420.0], "confidence": 0.92}
    helmet_b = {"bbox": [420.0, 50.0, 470.0, 90.0], "confidence": 0.89, "class_id": 1}

    tracked = tracker.update([person_a, person_b])
    assert len(tracked) == 2
    id_a = tracked[0]["worker_id"]
    id_b = tracked[1]["worker_id"]
    assert id_a != id_b
    assert "Worker " in id_a and "Worker " in id_b

    eval_a = verifier.evaluate_worker(
        worker_id=id_a,
        person_box=tracked[0]["bbox"],
        person_confidence=tracked[0]["confidence"],
        helmet_boxes=[helmet_a, helmet_b],
        cap_boxes=[],
        shoe_boxes=[shoe_a],
        frame_width=640,
        frame_height=480
    )

    eval_b = verifier.evaluate_worker(
        worker_id=id_b,
        person_box=tracked[1]["bbox"],
        person_confidence=tracked[1]["confidence"],
        helmet_boxes=[helmet_a, helmet_b],
        cap_boxes=[],
        shoe_boxes=[shoe_a],
        frame_width=640,
        frame_height=480
    )

    assert eval_a["status"] == SafetyStatus.SAFE
    assert eval_b["status"] == SafetyStatus.NOT_SAFE


def test_scenario_7_spatial_association_rejection(verifier):
    """Test 7: Safety equipment lying on the floor or background is NOT attributed to worker."""
    person_box = [200.0, 50.0, 360.0, 420.0]
    # Helmet lying on the ground (far away at x=50, y=400)
    stray_helmet = {"bbox": [40.0, 390.0, 80.0, 430.0], "confidence": 0.95, "class_id": 1}
    # Shoes lying on a table near the top (x=500, y=100)
    stray_shoes = {"bbox": [500.0, 90.0, 550.0, 130.0], "confidence": 0.90, "class_id": 3}

    res = verifier.evaluate_worker(
        worker_id="Worker 007",
        person_box=person_box,
        person_confidence=0.92,
        helmet_boxes=[stray_helmet],
        cap_boxes=[],
        shoe_boxes=[stray_shoes],
        frame_width=640,
        frame_height=480
    )

    # Both items should be rejected because they are not worn in head/feet zones!
    assert res["helmet_detected"] is False
    assert res["shoes_detected"] is False
    assert res["status"] == SafetyStatus.NOT_SAFE


def test_scenario_8_model_unavailable_diagnostics():
    """Test 8: AI model weights missing -> Transparent diagnostics, no fabricated predictions."""
    detector = SafetyDetector(model_path="non_existent_weights.pt")
    diag = detector.get_diagnostics()
    assert diag["is_custom_model"] is False
    assert "pending" in diag["status_message"].lower() or "custom ppe dataset training" in diag["status_message"].lower()

    # Inference on blank frame returns zero fake detections
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.detect(blank)
    assert len(res["persons"]) == 0
    assert len(res["helmets"]) == 0
    assert len(res["shoes"]) == 0
