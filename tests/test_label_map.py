"""Unit tests for the 7-class taxonomy and raw label mapping."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.label_map import (
    CLASSES,
    CLASS_TO_ID,
    class_role,
    make_raw_to_project,
    project_mask,
    raw_name_to_class,
)


def test_classes_integrity() -> None:
    expected = [
        "coral_alive", "coral_bleached", "coral_dead",
        "algae", "hard_substrate", "sand", "non_reef"
    ]
    assert CLASSES == expected
    assert len(CLASSES) == 7
    for i, name in enumerate(CLASSES):
        assert CLASS_TO_ID[name] == i


def test_raw_name_to_class_mapping() -> None:
    # Test key categories
    assert raw_name_to_class("Acropora alive") == "coral_alive"
    assert raw_name_to_class("acropora_bleached") == "coral_bleached"
    assert raw_name_to_class("brain coral dead") == "coral_dead"
    assert raw_name_to_class("algae covered substrate") == "algae"
    assert raw_name_to_class("hard substrate") == "hard_substrate"
    assert raw_name_to_class("sand") == "sand"
    assert raw_name_to_class("fish") == "non_reef"
    assert raw_name_to_class("background") == "non_reef"
    assert raw_name_to_class("unknown non-existent label") is None


def test_class_role() -> None:
    assert class_role("coral_alive") == "reef"
    assert class_role("coral_dead") == "reef"
    assert class_role("algae") == "reef"
    assert class_role("hard_substrate") == "reef"
    assert class_role("sand") == "reef"
    assert class_role("non_reef") == "non_reef"


def test_project_mask_mapping() -> None:
    raw_labels = {0: "unlabeled", 1: "acropora alive", 2: "sand", 3: "fish"}
    mapping = make_raw_to_project(raw_labels)
    assert mapping[0] is None
    assert mapping[1] == CLASS_TO_ID["coral_alive"]
    assert mapping[2] == CLASS_TO_ID["sand"]
    assert mapping[3] == CLASS_TO_ID["non_reef"]

    raw_mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    mapped = project_mask(raw_mask, mapping)

    assert mapped[0, 0] == 255  # Ignored / void
    assert mapped[0, 1] == CLASS_TO_ID["coral_alive"]
    assert mapped[1, 0] == CLASS_TO_ID["sand"]
    assert mapped[1, 1] == CLASS_TO_ID["non_reef"]


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} label-map tests passed")
