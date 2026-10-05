from __future__ import annotations

import json
from pathlib import Path

BROAD_CLASSES = [
    "coral_alive", "coral_bleached", "coral_dead", "algae",
    "rubble", "sand_rock", "other",
]
BROAD_TO_ID = {name: i for i, name in enumerate(BROAD_CLASSES)}


def load_coralscapes_classes(path: str | Path) -> dict[int, str]:
    raw = json.loads(Path(path).read_text())
    return {int(v): k for k, v in raw.items()}


def raw_name_to_broad(name: str) -> str | None:
    n = name.lower().strip()
    if n in {"background", "dark"}:
        return None
    if "algae" in n or n == "seagrass":
        return "algae"
    if n == "rubble":
        return "rubble"
    if n == "sand":
        return "sand_rock"
    if "bleached" in n:
        return "coral_bleached"
    if "dead" in n:
        return "coral_dead"
    if "alive" in n or "coral" in n:
        return "coral_alive"
    return "other"


def make_raw_to_broad(raw_classes: dict[int, str]) -> dict[int, int | None]:
    return {raw_id: (BROAD_TO_ID[broad] if (broad := raw_name_to_broad(name)) else None)
            for raw_id, name in raw_classes.items()}


def broad_mask(raw_mask, raw_to_broad):
    """Map an integer Coralscapes mask to broad IDs; 255 means ignored."""
    import numpy as np
    out = np.full(raw_mask.shape, 255, dtype=np.uint8)
    for raw_id, broad_id in raw_to_broad.items():
        if broad_id is not None:
            out[raw_mask == raw_id] = broad_id
    return out
