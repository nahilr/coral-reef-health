"""YOLO26s-sem training run for CoralscapesV2 on a Kaggle GPU kernel.

Downloads josauder/coralscapesV2 from Hugging Face, maps its 95 fine-grained
masks to the project taxonomy (``configs/labels.yaml`` + ``configs/label_map.json``),
trains YOLO26s-sem, and writes checkpoints + metrics into /kaggle/working for
download via:
    kaggle kernels output <user>/<slug> -p results/

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

import cv2
import numpy as np
import torch
from PIL import Image

print("STAGE pip-install-start", flush=True)
subprocess.run(
    [
        sys.executable, "-m", "pip", "install",
        "ultralytics==8.4.183", "datasets==4.8.5", "huggingface_hub==1.29.0",
    ],
    check=True,
)
print("STAGE pip-install-done", flush=True)
from datasets import load_dataset  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from ultralytics import YOLO  # noqa: E402

print("STAGE imports-done", flush=True)

OUT = Path("/kaggle/temp/coralscapes_v2_yolo26s_sem")
REPORT = Path("/kaggle/working/coralscapes_v2_yolo26s_sem_metrics.json")

# Project taxonomy: 7 taxonomy-free classes (configs/labels.yaml). The order
# here is the class-id order used by the model and by the inference pipeline.
CLASSES = [
    "coral_alive", "coral_bleached", "coral_dead",
    "algae", "hard_substrate", "sand", "non_reef",
]

# Source CoralscapesV2 label -> project class (configs/label_map.json).
RAW_TO_PROJECT = {
    "acanthaster planci": "non_reef",
    "acropora alive": "coral_alive",
    "acropora bleached": "coral_bleached",
    "acropora dead": "coral_dead",
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
    "brain coral dead": "coral_dead",
    "branching coral alive": "coral_alive",
    "branching coral bleached": "coral_bleached",
    "branching coral dead": "coral_dead",
    "cirrhipathes": "coral_alive",
    "dark": "non_reef",
    "dead giant clam": "non_reef",
    "feather worm": "non_reef",
    "fish": "non_reef",
    "fungiidae alive": "coral_alive",
    "fungiidae bleached": "coral_bleached",
    "fungiidae dead": "coral_dead",
    "galaxea alive": "coral_alive",
    "galaxea bleached": "coral_bleached",
    "galaxea dead": "coral_dead",
    "giant clam": "non_reef",
    "goniopora alive": "coral_alive",
    "goniopora dead": "coral_dead",
    "hard coral alive": "coral_alive",
    "hard coral bleached": "coral_bleached",
    "hard coral dead": "coral_dead",
    "hard substrate": "hard_substrate",
    "human": "non_reef",
    "human-made structure": "non_reef",
    "leather coral alive": "coral_alive",
    "lobophylliidae alive": "coral_alive",
    "lobophylliidae bleached": "coral_bleached",
    "lobophylliidae dead": "coral_dead",
    "massive alive": "coral_alive",
    "massive bleached": "coral_bleached",
    "massive dead": "coral_dead",
    "meandering coral alive": "coral_alive",
    "meandering coral bleached": "coral_bleached",
    "meandering coral dead": "coral_dead",
    "millepora": "coral_alive",
    "millepora bleached": "coral_bleached",
    "millepora dead": "coral_dead",
    "nephtheidae alive": "coral_alive",
    "other animal": "non_reef",
    "other coral alive": "coral_alive",
    "other coral bleached": "coral_bleached",
    "other coral dead": "coral_dead",
    "pavona alive": "coral_alive",
    "pavona bleached": "coral_bleached",
    "pavona dead": "coral_dead",
    "pocillopora alive": "coral_alive",
    "pocillopora bleached": "coral_bleached",
    "pocillopora dead": "coral_dead",
    "porites alive": "coral_alive",
    "porites bleached": "coral_bleached",
    "porites dead": "coral_dead",
    "rubble": "hard_substrate",
    "sand": "sand",
    "sea cucumber": "non_reef",
    "sea urchin": "non_reef",
    "seagrass": "non_reef",
    "seriatopora alive": "coral_alive",
    "seriatopora bleached": "coral_bleached",
    "seriatopora dead": "coral_dead",
    "soft coral (malacalcyonacea) alive": "coral_alive",
    "soft coral (malacalcyonacea) bleached": "coral_bleached",
    "sponge": "non_reef",
    "stylophora alive": "coral_alive",
    "stylophora bleached": "coral_bleached",
    "stylophora dead": "coral_dead",
    "table acropora alive": "coral_alive",
    "table acropora bleached": "coral_bleached",
    "table acropora dead": "coral_dead",
    "thin plate/encrusting coral alive": "coral_alive",
    "thin plate/encrusting coral bleached": "coral_bleached",
    "thin plate/encrusting coral dead": "coral_dead",
    "transect line": "non_reef",
    "transect tools": "non_reef",
    "trash": "non_reef",
    "turbinaria": "coral_alive",
    "turbinaria dead": "coral_dead",
    "water column": "non_reef",
    "water surface": "non_reef",
    "xeniidae alive": "coral_alive",
}

# The one-epoch smoke test completed successfully. This run trains for the
# requested 50 epochs, then evaluates once on the official test split.
SMOKE_TEST = False
EPOCHS = 1 if SMOKE_TEST else 50
IMGSZ = 1024
BATCH = 4
PATIENCE = 50
RUN_TEST_EVAL = not SMOKE_TEST


def project_class(name: str) -> int:
    """Map a source V2 label to a project class id. Raise if it is unmapped."""
    target = RAW_TO_PROJECT.get(name.lower().strip())
    if target is None:
        raise KeyError(f"Unmapped CoralscapesV2 label: {name!r}")
    return CLASSES.index(target)


id2label = json.loads(
    Path(hf_hub_download("josauder/coralscapesV2", "id2label.json", repo_type="dataset")).read_text()
)
raw_to_project = {int(k): project_class(v) for k, v in id2label.items()}
# Every class must be reachable; an empty class silently wastes a head slot.
_reached = {cid for cid in raw_to_project.values()}
if _reached != set(range(len(CLASSES))):
    raise RuntimeError(f"Project classes without source labels: {sorted(set(range(len(CLASSES))) - _reached)}")


counts = {}
for split in ["train", "validation", "test"]:
    print(f"STAGE convert-start {split}", flush=True)
    name = {"validation": "val"}.get(split, split)
    ds = load_dataset("josauder/coralscapesV2", split=split, streaming=True)
    (OUT / "images" / name).mkdir(parents=True, exist_ok=True)
    (OUT / "masks" / name).mkdir(parents=True, exist_ok=True)
    count = 0
    for idx, row in enumerate(ds):
        image = np.asarray(row["image"].convert("RGB"))
        raw = np.asarray(row["label"])
        mapped = np.full(raw.shape, 255, np.uint8)
        for rid, cid in raw_to_project.items():
            if cid is not None:
                mapped[raw == rid] = cid
        stem = f"{name}_{idx:06d}"
        Image.fromarray(image).save(OUT / "images" / name / f"{stem}.jpg", quality=95, optimize=True)
        Image.fromarray(mapped).save(OUT / "masks" / name / f"{stem}.png", compress_level=1)
        count += 1
        if count % 50 == 0:
            print(split, count, flush=True)
    counts[name] = count

# Fail fast (minutes in, not hours) if a split converted to nothing. The
# previous run trained for 4.6 h and then died because its test glob was empty.
for _name in ("train", "val", "test"):
    n_images = len(list((OUT / "images" / _name).glob("*.jpg")))
    n_masks = len(list((OUT / "masks" / _name).glob("*.png")))
    if n_images == 0 or n_images != n_masks:
        raise RuntimeError(
            f"Conversion produced {n_images} images / {n_masks} masks for split {_name!r}"
        )
print("STAGE conversion-verified", json.dumps(counts), flush=True)

(OUT / "data.yaml").write_text(json.dumps({
    "path": str(OUT), "train": "images/train", "val": "images/val", "test": "images/test",
    "masks_dir": "masks",
    "names": CLASSES,
}, indent=2))
(OUT / "label_map.json").write_text(json.dumps({
    "dataset": "josauder/coralscapesV2", "raw_classes": id2label,
    "raw_to_project": raw_to_project, "project_classes": CLASSES,
}, indent=2))
print(json.dumps({"counts": counts}, indent=2), flush=True)

model = YOLO("yolo26s-sem.pt")
results = model.train(
    data=str(OUT / "data.yaml"),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=0,
    workers=2,
    project="/kaggle/working/checkpoints",
    name="yolo26s_sem_coralscapes_v2",
    exist_ok=True,
    patience=PATIENCE,
    cache=False,
    plots=True,
    verbose=True,
)
save_dir = Path(getattr(results, "save_dir", None) or model.trainer.save_dir)
print(json.dumps({"save_dir": str(save_dir)}, indent=2))
best_checkpoint = save_dir / "weights" / "best.pt"
if not best_checkpoint.is_file():
    raise FileNotFoundError(f"Expected trained best checkpoint at {best_checkpoint}")

# The trainer still holds its model and optimizer on the GPU. Release them
# before loading best.pt for test inference, otherwise the second model can
# exhaust the 14.6 GB T4 (observed as a CUDA OOM on the seg-v2 run).
del model, results
gc.collect()
torch.cuda.empty_cache()
print("STAGE training-memory-released", round(torch.cuda.memory_allocated() / 1e9, 3), "GB", flush=True)
model = YOLO(str(best_checkpoint))


def evaluate_test_pixel_metrics(model):
    """Evaluate mapped test masks pixel-wise; undetected pixels count as false negatives."""
    nclasses = len(CLASSES)
    confusion = np.zeros((nclasses, nclasses + 1), dtype=np.int64)
    # The conversion loop writes RGB inputs as JPEG and masks as PNG.  Keep
    # the input glob aligned with those filenames so the official test split
    # is not silently evaluated as an empty set.
    test_images = sorted((OUT / "images" / "test").glob("*.jpg"))
    if not test_images:
        raise FileNotFoundError(f"No test images found under {OUT / 'images' / 'test'}")
    for image_path in test_images:
        # Predict one image per call. Passing the whole list to predict() makes
        # Ultralytics preload every 2048x1024 image, which can exhaust memory.
        prediction = model.predict(
            source=str(image_path), imgsz=IMGSZ,
            device=0, verbose=False,
        )[0]
        if prediction.semantic_mask is None:
            raise RuntimeError(f"No semantic mask returned for {image_path}")
        gt = np.asarray(Image.open(OUT / "masks" / "test" / f"{image_path.stem}.png"), dtype=np.uint8)
        h, w = gt.shape
        semantic = prediction.semantic_mask.data.detach().cpu().numpy()
        pred = semantic.astype(np.uint8)
        if pred.shape != (h, w):
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        pred[pred >= nclasses] = nclasses  # Reserve the extra column for invalid/unmodeled IDs.
        valid = gt < nclasses  # 255-mapped source labels stay ignored.
        bins = np.bincount(
            gt[valid].astype(np.int64) * (nclasses + 1) + pred[valid].astype(np.int64),
            minlength=nclasses * (nclasses + 1),
        ).reshape(nclasses, nclasses + 1)
        confusion += bins

    per_class = {}
    ious = []
    for cid, class_name in enumerate(CLASSES):
        tp = int(confusion[cid, cid])
        fn = int(confusion[cid].sum() - tp)
        fp = int(confusion[:, cid].sum() - tp)
        union = tp + fp + fn
        iou = tp / union if union else None
        per_class[class_name] = {
            "iou": iou, "pixel_tp": tp, "pixel_fp": fp, "pixel_fn": fn,
            "ground_truth_pixels": int(confusion[cid].sum()),
        }
        if iou is not None:
            ious.append(iou)
    report = {
        "dataset": "josauder/coralscapesV2",
        "split": "official test",
        "images": len(test_images),
        "classes": CLASSES,
        "pixel_accuracy": float(np.trace(confusion[:, :nclasses]) / max(1, confusion.sum())),
        "mean_iou": float(np.mean(ious)) if ious else None,
        "per_class": per_class,
        "notes": "YOLO26s-sem dense class-map metrics on mapped classes. Ground-truth source pixels mapped to 255 are excluded.",
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print("TEST_PIXEL_METRICS", json.dumps({
        "images": report["images"], "pixel_accuracy": report["pixel_accuracy"],
        "mean_iou": report["mean_iou"], "report": str(REPORT),
    }), flush=True)


if RUN_TEST_EVAL:
    evaluate_test_pixel_metrics(model)
else:
    print("STAGE smoke-test-complete; official test evaluation skipped", flush=True)
