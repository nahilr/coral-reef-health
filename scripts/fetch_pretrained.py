"""Download the two public pretrained Coralscapes backbones.

They are too large for git (~1.6 GB together), so only the fine-tuned
checkpoints in ``models/checkpoints`` are versioned (with Git LFS). This script
pulls the released EPFL-ECEO models into the paths the code expects and installs
the 95-class ``id2label.json`` sidecar that the DINOv3 repo does not ship.

    python scripts/fetch_pretrained.py                 # both
    python scripts/fetch_pretrained.py --only dinov3   # one

Requires network access and the ``huggingface_hub`` package (installed with
``transformers``). Both repos are public, so no token is needed.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRETRAINED = ROOT / "models" / "pretrained"
SIDECAR = ROOT / "configs" / "dinov3_id2label.json"

TARGETS = {
    "segformer": (
        "EPFL-ECEO/segformer-b5-finetuned-coralscapesv2-1024-1024-95_class",
        PRETRAINED / "segformer-b5-coralscapesv2-95",
    ),
    "dinov3": (
        "EPFL-ECEO/coralscapesv2-dinov3-vitl-lora-dpt-95_class",
        PRETRAINED / "coralscapesv2-dinov3-vitl-lora-dpt-95_class",
    ),
}


def fetch(repo_id: str, target: Path) -> None:
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {target}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        # Skip TensorFlow/Flax duplicates if a repo ever ships them.
        ignore_patterns=["*.h5", "*.msgpack", "*.onnx"],
    )
    weights = target / "model.safetensors"
    if not weights.is_file():
        raise RuntimeError(f"{repo_id} did not provide {weights.name} in {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=sorted(TARGETS),
        action="append",
        help="Fetch a single backbone (repeatable). Default: both.",
    )
    args = parser.parse_args()

    selected = args.only or list(TARGETS)
    if "dinov3" in selected and not SIDECAR.is_file():
        raise SystemExit(f"{SIDECAR} is missing; it is required for the 96->7 fold")

    for name in selected:
        repo_id, target = TARGETS[name]
        fetch(repo_id, target)

    if "dinov3" in selected:
        # The DINOv3 repo has num_classes=96 but no names; the fold is name-based.
        destination = TARGETS["dinov3"][1] / "id2label.json"
        shutil.copyfile(SIDECAR, destination)
        print(f"Installed {destination.relative_to(ROOT)}")

    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
