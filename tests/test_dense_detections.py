"""Regression tests for dense-backend region boxes and the reliability gate.

Covers the two bugs fixed together:

1. dense backends produced no detections and no confidence, so every image was
   forced into ``needs_review``;
2. dense backends drew no boxes at all.

Run directly (no test dependency):

    python tests/test_dense_detections.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.pipeline import detections_from_class_map
from src.trust.reliability import review_decision

CLASSES = ["coral_alive", "coral_bleached", "dead_coral", "algae",
           "hard_substrate", "sand", "non_reef"]


def synthetic_map() -> tuple[np.ndarray, np.ndarray]:
    """200x200 map: large live coral, bleached coral, small noise, big sand."""
    class_map = np.full((200, 200), 4, dtype=np.int16)  # hard_substrate
    confidence = np.full((200, 200), 0.9, dtype=np.float32)

    class_map[10:60, 10:60] = 0          # 2500 px live coral
    class_map[100:130, 100:140] = 1      # 1200 px bleached coral
    class_map[150:155, 150:155] = 2      # 25 px dead coral, below min area
    class_map[60:140, 10:90] = 5         # big sand area, must not get a box
    confidence[class_map == 0] = 0.8
    confidence[class_map == 1] = 0.6
    return class_map, confidence


def test_boxes_only_for_coral_classes() -> None:
    class_map, confidence = synthetic_map()
    dets = detections_from_class_map(class_map, confidence, CLASSES)
    assert {det["class"] for det in dets} == {"coral_alive", "coral_bleached"}


def test_boxes_are_sorted_largest_first_and_numbered() -> None:
    class_map, confidence = synthetic_map()
    dets = detections_from_class_map(class_map, confidence, CLASSES)
    assert [det["class"] for det in dets] == ["coral_alive", "coral_bleached"]
    assert [det["id"] for det in dets] == [0, 1]
    assert dets[0]["pixels"] >= dets[1]["pixels"]


def test_boxes_use_component_geometry() -> None:
    class_map, confidence = synthetic_map()
    dets = {det["class"]: det for det in detections_from_class_map(class_map, confidence, CLASSES)}
    assert dets["coral_alive"]["bbox"] == [10, 10, 60, 60]
    assert dets["coral_bleached"]["bbox"] == [100, 100, 140, 130]


def test_confidence_is_component_mean() -> None:
    class_map, confidence = synthetic_map()
    dets = {det["class"]: det for det in detections_from_class_map(class_map, confidence, CLASSES)}
    assert abs(dets["coral_alive"]["confidence"] - 0.8) < 1e-5
    assert abs(dets["coral_bleached"]["confidence"] - 0.6) < 1e-5


def test_small_components_are_dropped() -> None:
    class_map, confidence = synthetic_map()
    dets = detections_from_class_map(class_map, confidence, CLASSES)
    assert "dead_coral" not in {det["class"] for det in dets}


def test_max_detections_is_respected() -> None:
    class_map, confidence = synthetic_map()
    assert len(detections_from_class_map(class_map, confidence, CLASSES, max_detections=1)) == 1


def test_no_confidence_map_still_returns_boxes() -> None:
    class_map, _ = synthetic_map()
    dets = detections_from_class_map(class_map, None, CLASSES)
    assert dets and all(det["confidence"] is None for det in dets)


def test_gate_skips_unavailable_scores() -> None:
    decision = review_decision({"flags": []}, None, 0.9, None)
    assert decision["status"] == "accepted"
    assert decision["reliability_flags"] == []
    assert decision["confidence_available"] is False
    assert decision["uncertainty_available"] is False


def test_gate_still_flags_low_coverage_without_scores() -> None:
    decision = review_decision({"flags": []}, None, 0.01, None)
    assert decision["status"] == "needs_review"
    assert decision["reliability_flags"] == ["low_predicted_coverage"]


def test_gate_still_flags_real_low_confidence() -> None:
    decision = review_decision({"flags": []}, 0.10, 0.9, 0.1, {"confidence": 0.70})
    assert "low_confidence" in decision["reliability_flags"]
    decision = review_decision({"flags": []}, 0.9, 0.9, 0.8, {"uncertainty": 0.40})
    assert "high_uncertainty" in decision["reliability_flags"]


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} dense-detection tests passed")
