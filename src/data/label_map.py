"""Project taxonomy and explicit label mapping for Coralscapes.

The project taxonomy (``configs/labels.yaml``) collapses fine-grained Coralscapes
classes into 7 taxonomy-free task classes. The raw -> project mapping is declared
explicitly in ``configs/label_map.json`` so that every raw class has one reviewable
destination, instead of relying on substring heuristics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "configs" / "labels.yaml"
_MAPPING = _ROOT / "configs" / "label_map.json"

_CONFIG_DATA = yaml.safe_load(_CONFIG.read_text())
CLASSES = list(_CONFIG_DATA["classes"])
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

_ROLES = _CONFIG_DATA.get("roles", {})
REEF_CLASSES = list(_ROLES.get("reef", []))
EXCLUDED_FROM_HEALTH = set(_ROLES.get("excluded_from_health", []))
DISCARD_CLASSES = set(_ROLES.get("discard_sink", []))
IGNORE_ID = 255


def _normalise(name: str) -> str:
    """Normalise a raw label so lookups survive case/underscore/space drift."""
    return " ".join(name.lower().strip().replace("_", " ").split())


_RAW_NAME_TO_PROJECT: dict[str, str] = {
    _normalise(name): project
    for name, project in json.loads(_MAPPING.read_text()).items()
    if not name.startswith("_")
}


def raw_name_to_class(name: str) -> str | None:
    """Map a fine-grained raw label name to a project class, or None to ignore."""
    return _RAW_NAME_TO_PROJECT.get(_normalise(name))


def class_role(name: str) -> str:
    """Return a project class role: ``reef`` or ``non_reef``."""
    if name in REEF_CLASSES:
        return "reef"
    return "non_reef"


def load_raw_labels(path: str | Path) -> dict[int, str]:
    """Load raw class ID to name dictionary from JSON."""
    raw = json.loads(Path(path).read_text())
    return {int(class_id): str(name) for class_id, name in raw.items()}


def make_raw_to_project(raw_labels: dict[int, str]) -> dict[int, int | None]:
    """Map raw class IDs to project class IDs (or None if unmapped)."""
    return {
        raw_id: CLASS_TO_ID[project_name] if (project_name := raw_name_to_class(name)) else None
        for raw_id, name in raw_labels.items()
    }


def project_mask(raw_mask: np.ndarray, mapping: dict[int, int | None]) -> np.ndarray:
    """Map a raw integer mask to project classes; 255 is ignored."""
    mapped = np.full(raw_mask.shape, IGNORE_ID, dtype=np.uint8)
    for raw_id, project_id in mapping.items():
        if project_id is not None:
            mapped[raw_mask == raw_id] = project_id
    return mapped
