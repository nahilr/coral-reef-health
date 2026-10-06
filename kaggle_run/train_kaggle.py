"""Full Coralscapes training run for a Kaggle GPU kernel.

Downloads EPFL-ECEO/coralscapes from Hugging Face, converts masks to the
project's 7-class YOLO segmentation format, trains YOLO26n-seg, and writes
checkpoints + metrics into /kaggle/working for download via:
    kaggle kernels output <user>/<slug> -p results/
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

print("STAGE pip-install-start", flush=True)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "ultralytics", "datasets", "huggingface_hub"],
    check=True,
)
print("STAGE pip-install-done", flush=True)
from datasets import load_dataset  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from ultralytics import YOLO  # noqa: E402

print("STAGE imports-done", flush=True)

OUT = Path("/kaggle/working/coralscapes_yolo_hf")
CLASSES = ["coral_alive", "coral_bleached", "coral_dead", "algae", "rubble", "sand_rock", "other"]
EPOCHS = 30
IMGSZ = 512
BATCH = 16


def broad(name: str):
    n = name.lower().strip()
    if n in {"background", "dark"}:
        return None
    if n == "dead clam":
        return 6
    if "algae" in n or n == "seagrass":
        return 3
    if n == "rubble":
        return 4
    if n == "sand":
        return 5
    if "bleached" in n:
        return 1
    if "dead" in n:
        return 2
    if "alive" in n or "coral" in n or "millepora" in n or "turbinaria" in n:
        return 0
    return 6


id2label = json.loads(
    Path(hf_hub_download("EPFL-ECEO/coralscapes", "id2label.json", repo_type="dataset")).read_text()
)
raw_to_broad = {int(k): broad(v) for k, v in id2label.items()}


def contours_to_yolo(mask, cid, out_size=(1024, 512), min_area=20):
    resized = cv2.resize(mask, out_size, interpolation=cv2.INTER_NEAREST)
    binary = (resized == cid).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    width, height = out_size
    for c in contours:
        if cv2.contourArea(c) < min_area or len(c) < 3:
            continue
        c = cv2.approxPolyDP(c, max(1.0, 0.002 * cv2.arcLength(c, True)), True).reshape(-1, 2)
        if len(c) < 3:
            continue
        pts = []
        for x, y in c:
            pts += [f"{float(x)/width:.6f}", f"{float(y)/height:.6f}"]
        lines.append(f"{cid} " + " ".join(pts))
    return lines


counts = {}
for split in ["train", "validation", "test"]:
    print(f"STAGE convert-start {split}", flush=True)
    name = {"validation": "val"}.get(split, split)
    ds = load_dataset("EPFL-ECEO/coralscapes", split=split, streaming=True)
    (OUT / "images" / name).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / name).mkdir(parents=True, exist_ok=True)
    count = 0
    for idx, row in enumerate(ds):
        image = np.asarray(row["image"].convert("RGB"))
        raw = np.asarray(row["label"])
        mapped = np.full(raw.shape, 255, np.uint8)
        for rid, cid in raw_to_broad.items():
            if cid is not None:
                mapped[raw == rid] = cid
        lines = []
        for cid in range(len(CLASSES)):
            lines.extend(contours_to_yolo(mapped, cid))
        stem = f"{name}_{idx:06d}"
        Image.fromarray(image).save(OUT / "images" / name / f"{stem}.png", compress_level=1)
        (OUT / "labels" / name / f"{stem}.txt").write_text(("\n".join(lines) + "\n") if lines else "")
        count += 1
        if count % 50 == 0:
            print(split, count, flush=True)
    counts[name] = count

(OUT / "data.yaml").write_text(json.dumps({
    "path": str(OUT), "train": "images/train", "val": "images/val", "test": "images/test",
    "names": {i: n for i, n in enumerate(CLASSES)},
}, indent=2))
print(json.dumps({"counts": counts}, indent=2), flush=True)

model = YOLO("yolo26n-seg.pt")
results = model.train(
    data=str(OUT / "data.yaml"),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=0,
    workers=2,
    project="/kaggle/working/checkpoints",
    name="yolo26n_full",
    exist_ok=True,
    patience=10,
    cache=False,
    plots=True,
    verbose=True,
)
print(json.dumps({"save_dir": str(results.save_dir)}, indent=2))
