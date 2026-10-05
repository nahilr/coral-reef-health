from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .label_map import BROAD_CLASSES, load_coralscapes_classes, make_raw_to_broad


def contours_to_yolo(mask: np.ndarray, broad_id: int, output_size=(1024, 512), min_area=20):
    resized = cv2.resize(mask, output_size, interpolation=cv2.INTER_NEAREST)
    binary = (resized == broad_id).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    width, height = output_size
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
        lines.append(f"{broad_id} " + " ".join(points))
    return lines


def prepare(root: Path, out: Path, max_images_per_split: int | None = None,
            copy_images: bool = False) -> dict:
    root, out = root.resolve(), out.resolve()
    if (root / "id2label.json").exists():
        return prepare_from_hf_snapshot(root, out, max_images_per_split=max_images_per_split,
                                        copy_images=copy_images)
    raw_classes = load_coralscapes_classes(root / "classes.json")
    raw_to_broad = make_raw_to_broad(raw_classes)
    rows = []
    for split in ("train", "val", "test"):
        source = root / "leftImg8bit" / split
        images = sorted(source.glob("site*/*_leftImg8bit.png"))
        if max_images_per_split:
            images = images[:max_images_per_split]
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image_path in images:
            relative = image_path.relative_to(source)
            mask_path = root / "gtFine" / split / relative.parent / relative.name.replace("_leftImg8bit.png", "_gtFine.png")
            if not mask_path.exists():
                continue
            target_image = out / "images" / split / relative
            target_label = out / "labels" / split / relative.with_suffix(".txt")
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            if not target_image.exists():
                if copy_images:
                    shutil.copy2(image_path, target_image)
                else:
                    os.symlink(image_path, target_image)
            raw = np.asarray(Image.open(mask_path))
            mapped = np.full(raw.shape, 255, dtype=np.uint8)
            for raw_id, broad_id in raw_to_broad.items():
                if broad_id is not None:
                    mapped[raw == raw_id] = broad_id
            lines = []
            for broad_id in range(len(BROAD_CLASSES)):
                lines.extend(contours_to_yolo(mapped, broad_id))
            target_label.write_text("\n".join(lines) + ("\n" if lines else ""))
            rows.append({"image": str(target_image), "mask": str(mask_path), "split": split,
                         "site": relative.parts[0], "group": relative.stem.rsplit("_", 2)[0],
                         "label_count": len(lines)})
    manifest = out / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["image"])
        writer.writeheader(); writer.writerows(rows)
    config = {"path": str(out), "train": "images/train", "val": "images/val",
              "test": "images/test", "names": {i: name for i, name in enumerate(BROAD_CLASSES)}}
    (out / "data.yaml").write_text(json.dumps(config, indent=2))
    (out / "label_map.json").write_text(json.dumps({"raw_classes": raw_classes,
        "raw_to_broad": raw_to_broad, "broad_classes": BROAD_CLASSES}, indent=2))
    return {"images": len(rows), "by_split": {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}, "out": str(out)}


def prepare_from_hf_snapshot(root: Path, out: Path, max_images_per_split: int | None = None,
                             copy_images: bool = False) -> dict:
    """Convert a local HF snapshot of EPFL-ECEO/coralscapes into the same
    YOLO segmentation dataset format as the Zenodo-based prepare()."""
    import pyarrow.parquet as pq

    id2label = json.loads((root / "id2label.json").read_text())
    raw_classes = {int(i): n for i, n in id2label.items()}
    classes_name_to_id = {v: k for k, v in raw_classes.items()}
    raw_to_broad = make_raw_to_broad(raw_classes)
    name_map = {"train": "train", "validation": "val", "test": "test"}
    rows = []
    import io
    for split, out_split in name_map.items():
        files = sorted((root / "data").glob(f"{split}-*"))
        if not files:
            continue
        (out / "images" / out_split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / out_split).mkdir(parents=True, exist_ok=True)
        (out / "masks" / out_split).mkdir(parents=True, exist_ok=True)
        split_index = 0
        rows_for_split = 0
        for f in files:
            table = pq.read_table(str(f))
            for row in table.to_pylist():
                if max_images_per_split and rows_for_split >= max_images_per_split:
                    break
                image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                raw = np.asarray(Image.open(io.BytesIO(row["label"]["bytes"])))
                stem = f"{out_split}_{split_index:06d}"
                split_index += 1
                image_path = out / "images" / out_split / f"{stem}.png"
                mask_path = out / "masks" / out_split / f"{stem}.png"
                label_path = out / "labels" / out_split / f"{stem}.txt"
                image.save(image_path, compress_level=1)
                Image.fromarray(raw).save(mask_path)
                mapped = np.full(raw.shape, 255, dtype=np.uint8)
                for raw_id, broad_id in raw_to_broad.items():
                    if broad_id is not None:
                        mapped[raw == raw_id] = broad_id
                lines = []
                for broad_id in range(len(BROAD_CLASSES)):
                    lines.extend(contours_to_yolo(mapped, broad_id))
                label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
                rows.append({"image": str(image_path), "mask": str(mask_path), "split": out_split,
                             "site": "", "group": stem, "label_count": len(lines)})
                rows_for_split += 1
    manifest = out / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["image"])
        writer.writeheader(); writer.writerows(rows)
    config = {"path": str(out), "train": "images/train", "val": "images/val",
              "test": "images/test", "names": {i: name for i, name in enumerate(BROAD_CLASSES)}}
    (out / "data.yaml").write_text(json.dumps(config, indent=2))
    (out / "label_map.json").write_text(json.dumps({"raw_classes": raw_classes,
        "raw_to_broad": raw_to_broad, "broad_classes": BROAD_CLASSES}, indent=2))
    (out / "classes.json").write_text(json.dumps(classes_name_to_id, indent=2))
    return {"images": len(rows), "by_split": {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}, "out": str(out)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("datasets/coralscapes_hf"))
    p.add_argument("--out", type=Path, default=Path("data/processed/coralscapes_yolo"))
    p.add_argument("--max-images-per-split", type=int, default=None)
    p.add_argument("--copy-images", action="store_true",
                   help="Copy image files instead of creating local symlinks (portable for Colab/Drive).")
    args = p.parse_args()
    print(json.dumps(prepare(args.root, args.out, args.max_images_per_split, args.copy_images), indent=2))


if __name__ == "__main__":
    main()
