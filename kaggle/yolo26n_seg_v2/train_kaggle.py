"""Train YOLO26n-seg on CoralscapesV2 with component-derived instances.

CoralscapesV2 supplies semantic coral masks, not per-colony coral instance IDs.
This experiment turns connected components within each mapped semantic class
into YOLO segmentation polygons. Its box/mask AP is therefore against those
component instances and must not be described as true colony-detection accuracy.

A Kaggle kernel has no copy of the repository, so the source-label -> project
mapping is inlined below. Keep it byte-identical to ``configs/label_map.json``;
the script aborts if any CoralscapesV2 label is unmapped.
"""
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

# Set before torch initialises CUDA: the previous run OOM'd in the test phase.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

print("STAGE pip-install-start", flush=True)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "ultralytics", "datasets", "huggingface_hub", "opencv-python-headless"],
    check=True,
)
print("STAGE pip-install-done", flush=True)

import cv2
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO

# Project taxonomy: 7 taxonomy-free classes (configs/labels.yaml).
CLASSES = [
    "coral_alive", "coral_bleached", "dead_coral",
    "algae", "hard_substrate", "sand", "non_reef",
]
NCLASS = len(CLASSES)
EPOCHS, IMGSZ, BATCH = 30, 1024, 4
ROOT = Path("/kaggle/temp/coralscapes_v2_yolo_seg")
RUNS = Path("/kaggle/working/checkpoints")
REPORT = Path("/kaggle/working/coralscapes_v2_yolo_seg_metrics.json")

# Source CoralscapesV2 label -> project class (configs/label_map.json).
RAW_TO_PROJECT = {
    "acanthaster planci": "non_reef",
    "acropora alive": "coral_alive",
    "acropora bleached": "coral_bleached",
    "acropora dead": "dead_coral",
    "algae covered brain coral": "algae",
    "algae covered branching coral": "algae",
    "algae covered lobophylliidae": "algae",
    "algae covered massive coral": "algae",
    "algae covered pocillopora": "algae",
    "algae covered porites": "algae",
    "algae covered substrate": "algae",
    "algae covered table acropora": "algae",
    "anemone": "non_reef",
    "background": "non_reef",
    "brain coral alive": "coral_alive",
    "brain coral bleached": "coral_bleached",
    "brain coral dead": "dead_coral",
    "branching coral alive": "coral_alive",
    "branching coral bleached": "coral_bleached",
    "branching coral dead": "dead_coral",
    "cirrhipathes": "coral_alive",
    "dark": "non_reef",
    "dead giant clam": "non_reef",
    "feather worm": "non_reef",
    "fish": "non_reef",
    "fungiidae alive": "coral_alive",
    "fungiidae bleached": "coral_bleached",
    "fungiidae dead": "dead_coral",
    "galaxea alive": "coral_alive",
    "galaxea bleached": "coral_bleached",
    "galaxea dead": "dead_coral",
    "giant clam": "non_reef",
    "goniopora alive": "coral_alive",
    "goniopora dead": "dead_coral",
    "hard coral alive": "coral_alive",
    "hard coral bleached": "coral_bleached",
    "hard coral dead": "dead_coral",
    "hard substrate": "hard_substrate",
    "human": "non_reef",
    "human-made structure": "non_reef",
    "leather coral alive": "coral_alive",
    "lobophylliidae alive": "coral_alive",
    "lobophylliidae bleached": "coral_bleached",
    "lobophylliidae dead": "dead_coral",
    "massive alive": "coral_alive",
    "massive bleached": "coral_bleached",
    "massive dead": "dead_coral",
    "meandering coral alive": "coral_alive",
    "meandering coral bleached": "coral_bleached",
    "meandering coral dead": "dead_coral",
    "millepora": "coral_alive",
    "millepora bleached": "coral_bleached",
    "millepora dead": "dead_coral",
    "nephtheidae alive": "coral_alive",
    "other animal": "non_reef",
    "other coral alive": "coral_alive",
    "other coral bleached": "coral_bleached",
    "other coral dead": "dead_coral",
    "pavona alive": "coral_alive",
    "pavona bleached": "coral_bleached",
    "pavona dead": "dead_coral",
    "pocillopora alive": "coral_alive",
    "pocillopora bleached": "coral_bleached",
    "pocillopora dead": "dead_coral",
    "porites alive": "coral_alive",
    "porites bleached": "coral_bleached",
    "porites dead": "dead_coral",
    "rubble": "hard_substrate",
    "sand": "sand",
    "sea cucumber": "non_reef",
    "sea urchin": "non_reef",
    "seagrass": "non_reef",
    "seriatopora alive": "coral_alive",
    "seriatopora bleached": "coral_bleached",
    "seriatopora dead": "dead_coral",
    "soft coral (malacalcyonacea) alive": "coral_alive",
    "soft coral (malacalcyonacea) bleached": "coral_bleached",
    "sponge": "non_reef",
    "stylophora alive": "coral_alive",
    "stylophora bleached": "coral_bleached",
    "stylophora dead": "dead_coral",
    "table acropora alive": "coral_alive",
    "table acropora bleached": "coral_bleached",
    "table acropora dead": "dead_coral",
    "thin plate/encrusting coral alive": "coral_alive",
    "thin plate/encrusting coral bleached": "coral_bleached",
    "thin plate/encrusting coral dead": "dead_coral",
    "transect line": "non_reef",
    "transect tools": "non_reef",
    "trash": "non_reef",
    "turbinaria": "coral_alive",
    "turbinaria dead": "dead_coral",
    "water column": "non_reef",
    "water surface": "non_reef",
    "xeniidae alive": "coral_alive",
}


def project_class(name: str) -> int:
    """Map a source V2 label to a project class id. Raise if it is unmapped."""
    target = RAW_TO_PROJECT.get(name.lower().strip())
    if target is None:
        raise KeyError(f"Unmapped CoralscapesV2 label: {name!r}")
    return CLASSES.index(target)


def polygon_labels(mask, min_area=20):
    """Export connected semantic components as YOLO-seg instances.

    Component-derived instances are region-level for 'stuff' classes (sand,
    hard substrate, rubble, non_reef), which the contour pass returns as single
    large polygons with any holes ignored.
    """
    h, w = mask.shape
    # Half-resolution contours reduce label size and remove annotation noise.
    small = cv2.resize(mask, (max(64, w // 2), max(32, h // 2)), interpolation=cv2.INTER_NEAREST)
    sh, sw = small.shape
    lines = []
    for cid in range(NCLASS):
        binary = (small == cid).astype(np.uint8)
        components, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in components:
            if cv2.contourArea(contour) < min_area or len(contour) < 3:
                continue
            contour = cv2.approxPolyDP(contour, max(1.0, .002 * cv2.arcLength(contour, True)), True)
            points = contour.reshape(-1, 2)
            if len(points) < 3:
                continue
            coords = " ".join(f"{x / sw:.6f} {y / sh:.6f}" for x, y in points)
            lines.append(f"{cid} {coords}")
    return lines


print("STAGE dataset-conversion-start", flush=True)
id2label_path = hf_hub_download("josauder/coralscapesV2", "id2label.json", repo_type="dataset")
id2label = json.loads(Path(id2label_path).read_text())
raw_to_project = {int(rid): project_class(name) for rid, name in id2label.items()}
# Every class must be reachable; an empty class silently wastes a head slot.
_reached = set(raw_to_project.values())
if _reached != set(range(NCLASS)):
    raise RuntimeError(f"Project classes without source labels: {sorted(set(range(NCLASS)) - _reached)}")
split_counts = {}
instance_counts = {}

for split, out_split in (("train", "train"), ("validation", "val"), ("test", "test")):
    for kind in ("images", "labels", "masks"):
        (ROOT / kind / out_split).mkdir(parents=True, exist_ok=True)
    ds = load_dataset("josauder/coralscapesV2", split=split, streaming=True)
    image_count = 0
    for idx, row in enumerate(ds):
        image = np.asarray(row["image"].convert("RGB"))
        raw = np.asarray(row["label"])
        mapped = np.full(raw.shape, 255, dtype=np.uint8)
        for rid, cid in raw_to_project.items():
            if cid is not None:
                mapped[raw == rid] = cid
        stem = f"{out_split}_{idx:06d}"
        Image.fromarray(image).save(ROOT / "images" / out_split / f"{stem}.jpg", quality=95)
        Image.fromarray(mapped).save(ROOT / "masks" / out_split / f"{stem}.png", compress_level=1)
        lines = polygon_labels(mapped)
        (ROOT / "labels" / out_split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        instance_counts[out_split] = instance_counts.get(out_split, 0) + len(lines)
        image_count += 1
        if image_count % 50 == 0:
            print(f"STAGE converted {split} {image_count} images", flush=True)
    split_counts[out_split] = image_count

# Fail fast (minutes in, not hours) if a split converted to nothing, or if the
# image/mask/label stems would not line up for training and evaluation.
for _name in ("train", "val", "test"):
    n_images = len(list((ROOT / "images" / _name).glob("*.jpg")))
    n_masks = len(list((ROOT / "masks" / _name).glob("*.png")))
    n_labels = len(list((ROOT / "labels" / _name).glob("*.txt")))
    if n_images == 0 or n_images != n_masks or n_images != n_labels:
        raise RuntimeError(
            f"Conversion produced {n_images} images / {n_masks} masks / {n_labels} labels "
            f"for split {_name!r}"
        )
print("STAGE conversion-verified", json.dumps(split_counts), flush=True)

data = {"path": str(ROOT), "train": "images/train", "val": "images/val", "test": "images/test",
        "names": {i: name for i, name in enumerate(CLASSES)}}
(ROOT / "data.yaml").write_text(json.dumps(data, indent=2))
(ROOT / "label_map.json").write_text(json.dumps({
    "dataset": "josauder/coralscapesV2", "classes": CLASSES, "raw_to_project": raw_to_project,
    "instance_annotation_method": "connected components of mapped semantic masks; component-derived instances, not coral colony instance labels",
}, indent=2))
print("DATASET_COUNTS", json.dumps({"images": split_counts, "instances": instance_counts}), flush=True)

print("STAGE training-start", flush=True)
model = YOLO("yolo26n-seg.pt")
model.train(
    data=str(ROOT / "data.yaml"), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
    device=0, workers=2, project=str(RUNS), name="yolo26n_seg_coralscapes_v2",
    exist_ok=True, patience=10, cache=False, plots=True, verbose=True,
)
save_dir = Path(model.trainer.save_dir)
best = save_dir / "weights" / "best.pt"
if not best.is_file():
    raise FileNotFoundError(f"Missing trained checkpoint: {best}")
# results_dict values are normally scalars, but coerce defensively: this dict
# is embedded in json.dumps at the very end of the run, and a serialisation
# error there would discard the test metrics we just computed.
try:
    val_metrics = {
        str(key): float(value)
        for key, value in (getattr(model.metrics, "results_dict", {}) or {}).items()
    }
except Exception as exc:  # pragma: no cover - never lose the test metrics over this
    print("WARNING could not serialise ultralytics validation metrics:", exc, flush=True)
    val_metrics = {}
print("STAGE training-done", str(save_dir), flush=True)

# Release the trainer, model and optimizer before loading best.pt for test
# inference. Keeping both resident caused the CUDA OOM in the previous run.
del model
gc.collect()
torch.cuda.empty_cache()
print("STAGE training-memory-released", round(torch.cuda.memory_allocated() / 1e9, 3), "GB", flush=True)


def evaluate_pixel_metrics(checkpoint):
    """Rasterize instance predictions and score them against V2 semantic masks."""
    yolo = YOLO(str(checkpoint))
    confusion = np.zeros((NCLASS, NCLASS + 1), dtype=np.int64)
    test_images = sorted((ROOT / "images" / "test").glob("*.jpg"))
    if not test_images:
        raise FileNotFoundError(f"No test images found under {ROOT / 'images' / 'test'}")
    for image_path in test_images:
        # One image per call; a full list makes Ultralytics preload the split.
        result = yolo.predict(
            source=str(image_path), imgsz=IMGSZ,
            device=0, verbose=False,
        )[0]
        gt = np.asarray(Image.open(ROOT / "masks" / "test" / f"{image_path.stem}.png"), dtype=np.uint8)
        h, w = gt.shape
        pred = np.full((h, w), NCLASS, dtype=np.uint8)
        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            scores = result.boxes.conf.detach().cpu().numpy()
            # Draw low confidence first so the highest-confidence mask wins overlaps.
            for idx in np.argsort(scores):
                cls = int(classes[idx])
                if cls < 0 or cls >= NCLASS:  # Never write an out-of-range class id.
                    continue
                mask = cv2.resize(masks[idx], (w, h), interpolation=cv2.INTER_NEAREST) > .5
                pred[mask] = cls
        valid = gt < NCLASS
        bins = np.bincount(
            gt[valid].astype(np.int64) * (NCLASS + 1) + pred[valid].astype(np.int64),
            minlength=NCLASS * (NCLASS + 1),
        ).reshape(NCLASS, NCLASS + 1)
        confusion[:] += bins
    ious, per_class = [], {}
    for cid, name in enumerate(CLASSES):
        tp = int(confusion[cid, cid])
        fn = int(confusion[cid].sum() - tp)
        fp = int(confusion[:, cid].sum() - tp)
        union = tp + fp + fn
        iou = tp / union if union else None
        per_class[name] = {"iou": iou, "tp": tp, "fp": fp, "fn": fn}
        if iou is not None:
            ious.append(iou)
    return {
        "images": len(test_images),
        "pixel_accuracy": float(np.trace(confusion[:, :NCLASS]) / max(1, confusion.sum())),
        "mean_iou": float(np.mean(ious)) if ious else None,
        "per_class": per_class,
    }


print("STAGE test-evaluation-start", flush=True)
pixel_metrics = evaluate_pixel_metrics(best)
summary = {
    "dataset": "josauder/coralscapesV2", "split": "official test",
    "training": {"architecture": "YOLO26n-seg", "epochs": EPOCHS, "imgsz": IMGSZ, "batch": BATCH,
                  "instance_counts": instance_counts},
    "validation_note": "YOLO mask AP is measured against connected-component instances derived from semantic labels; not true colony-instance accuracy.",
    "pixel_metrics": pixel_metrics,
    "best_checkpoint": str(best),
    "ultralytics_validation_metrics": val_metrics,
}
REPORT.write_text(json.dumps(summary, indent=2, default=float))
print("TEST_PIXEL_METRICS", json.dumps({k: pixel_metrics[k] for k in ("images", "pixel_accuracy", "mean_iou")}), flush=True)
print("TRAINING_REPORT", str(REPORT), flush=True)
