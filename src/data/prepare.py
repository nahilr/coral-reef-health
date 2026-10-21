"""Prepare Coralscapes dataset into the project 7-class YOLO segmentation format."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pyarrow.parquet as pq

from .label_map import (
    CLASSES,
    load_raw_labels,
    make_raw_to_project,
    project_mask,
)


def contours_to_yolo(mask: np.ndarray, class_id: int, output_size=None, min_area=20):
    """Convert binary class mask contours into YOLO normalized polygon annotations."""
    if output_size is not None:
        resized = cv2.resize(mask, output_size, interpolation=cv2.INTER_NEAREST)
        width, height = output_size
    else:
        h, w = mask.shape[:2]
        width, height = max(64, w // 2), max(32, h // 2)
        resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    binary = (resized == class_id).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area or len(contour) < 3:
            continue
        epsilon = max(1.0, 0.002 * cv2.arcLength(contour, True))
        contour = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(contour) < 3:
            continue
        points = []
        for x, y in contour:
            points.extend([f"{float(x) / width:.6f}", f"{float(y) / height:.6f}"])
        lines.append(f"{class_id} " + " ".join(points))
    return lines


def prepare(root: Path, out: Path, max_images_per_split: int | None = None) -> dict:
    """Prepare the Coralscapes snapshot as project-taxonomy YOLO data."""
    root, out = root.resolve(), out.resolve()
    id2label_file = root / "id2label.json"
    if not id2label_file.is_file():
        # Fall back to root/configs if present
        id2label_file = Path(__file__).resolve().parents[2] / "configs" / "dinov3_id2label.json"
    raw_labels = load_raw_labels(id2label_file)
    raw_to_project = make_raw_to_project(raw_labels)
    rows = []
    split_names = {"train": "train", "validation": "val", "test": "test"}

    for split, out_split in split_names.items():
        parquet_files = sorted((root / "data").glob(f"{split}-*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"Missing required Coralscapes {split} Parquet files under {root / 'data'}"
            )
        for subdir in ("images", "masks", "labels"):
            (out / subdir / out_split).mkdir(parents=True, exist_ok=True)

        index = 0
        for parquet_path in parquet_files:
            parquet = pq.ParquetFile(parquet_path)
            for batch in parquet.iter_batches(batch_size=8):
                for row in batch.to_pylist():
                    if max_images_per_split is not None and index >= max_images_per_split:
                        break
                    image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                    raw_mask = np.asarray(Image.open(io.BytesIO(row["label"]["bytes"])))
                    mapped = project_mask(raw_mask, raw_to_project)
                    stem = f"{out_split}_{index:06d}"
                    image_path = out / "images" / out_split / f"{stem}.png"
                    mask_path = out / "masks" / out_split / f"{stem}.png"
                    label_path = out / "labels" / out_split / f"{stem}.txt"
                    image.save(image_path, compress_level=1)
                    Image.fromarray(raw_mask.astype(np.uint8)).save(mask_path)
                    yolo_lines = []
                    for class_id in range(len(CLASSES)):
                        yolo_lines.extend(contours_to_yolo(mapped, class_id))
                    label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
                    rows.append({
                        "image": str(image_path), "mask": str(mask_path), "split": out_split,
                        "site": str(row.get("site", "")), "group": str(row.get("stem", stem)),
                        "label_count": len(yolo_lines),
                    })
                    index += 1
                if max_images_per_split is not None and index >= max_images_per_split:
                    break
        print(f"Prepared Coralscapes {split}: {index} images", flush=True)

    if not rows:
        raise RuntimeError(f"No images were prepared from {root}")
    with (out / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    config = {
        "path": str(out), "train": "images/train", "val": "images/val", "test": "images/test",
        "masks_dir": "masks",
        "names": CLASSES,
    }
    (out / "data.yaml").write_text(json.dumps(config, indent=2))
    (out / "classes.json").write_text(json.dumps({i: name for i, name in raw_labels.items()}, indent=2))
    (out / "label_map.json").write_text(json.dumps({
        "raw_classes": raw_labels,
        "raw_to_project": raw_to_project,
        "project_classes": CLASSES,
        "ignored_raw_classes": [name for raw_id, name in raw_labels.items() if raw_to_project[raw_id] is None],
    }, indent=2))
    return {
        "images": len(rows),
        "by_split": {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")},
        "classes": CLASSES, "out": str(out),
    }

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/raw/coralscapes"))
    p.add_argument("--out", type=Path, default=Path("data/processed/yolo"))
    p.add_argument("--max-images-per-split", type=int, default=None)
    args = p.parse_args()
    print(json.dumps(prepare(args.root, args.out, args.max_images_per_split), indent=2))


if __name__ == "__main__":
    main()
