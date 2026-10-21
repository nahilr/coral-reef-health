"""Project taxonomy and explicit label mapping for CoralscapesV2.

The project taxonomy (``configs/labels.yaml``) collapses the 95 fine-grained
CoralscapesV2 classes into 7 taxonomy-free task classes. The V2 -> project
mapping is declared explicitly in ``configs/label_map.json`` so that every
V2 class has one reviewable destination, instead of relying on substring
heuristics that drift as labels change.
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
V2_CLASSES = list(_CONFIG_DATA["classes"])
V2_TO_ID = {name: i for i, name in enumerate(V2_CLASSES)}

_ROLES = _CONFIG_DATA.get("roles", {})
REEF_CLASSES = list(_ROLES.get("reef", []))
EXCLUDED_FROM_HEALTH = set(_ROLES.get("excluded_from_health", []))
DISCARD_CLASSES = set(_ROLES.get("discard_sink", []))
IGNORE_ID = 255


def _normalise(name: str) -> str:
    """Normalise a raw label so lookups survive case/underscore/space drift."""
    return " ".join(name.lower().strip().replace("_", " ").split())


_V2_NAME_TO_PROJECT: dict[str, str] = {
    _normalise(name): project
    for name, project in json.loads(_MAPPING.read_text()).items()
    if not name.startswith("_")
}


def v2_name_to_class(name: str) -> str | None:
    """Map a fine-grained V2 label name to a project class, or None to ignore."""
    return _V2_NAME_TO_PROJECT.get(_normalise(name))


def class_role(name: str) -> str:
    """Return a project class role: ``reef`` or ``non_reef``."""
    if name in REEF_CLASSES:
        return "reef"
    return "non_reef"


def load_v2_labels(path: str | Path) -> dict[int, str]:
    raw = json.loads(Path(path).read_text())
    return {int(class_id): str(name) for class_id, name in raw.items()}


def make_v2_to_project(raw_labels: dict[int, str]) -> dict[int, int | None]:
    return {
        raw_id: V2_TO_ID[project_name] if (project_name := v2_name_to_class(name)) else None
        for raw_id, name in raw_labels.items()
    }


def project_v2_mask(raw_mask: np.ndarray, mapping: dict[int, int | None]) -> np.ndarray:
    """Map a V2 integer mask to the project classes; 255 is ignored."""
    mapped = np.full(raw_mask.shape, IGNORE_ID, dtype=np.uint8)
    for raw_id, project_id in mapping.items():
        if project_id is not None:
            mapped[raw_mask == raw_id] = project_id
    return mapped
