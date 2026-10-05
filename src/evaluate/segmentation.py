from __future__ import annotations

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.data.label_map import BROAD_CLASSES, broad_mask, load_coralscapes_classes, make_raw_to_broad


def predicted_map(result, shape):
    h, w = shape; out = np.full((h, w), 255, dtype=np.uint8); conf = np.full((h, w), -1.0, dtype=np.float32)
    if result.masks is None or result.boxes is None: return out
    masks = result.masks.data.detach().cpu().numpy(); classes = result.boxes.cls.detach().cpu().numpy().astype(int); scores = result.boxes.conf.detach().cpu().numpy()
    for mask, cls, score in zip(masks, classes, scores):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
        sel = mask & (score > conf); out[sel], conf[sel] = cls, score
    return out


def evaluate(model_path: str | Path, manifest: str | Path, raw_classes_path: str | Path,
             max_images: int | None = None, conf: float = 0.20, imgsz: int = 384):
    df = pd.read_csv(manifest); df = df[df.split == "test"]
    if max_images: df = df.head(max_images)
    model = YOLO(str(model_path)); raw_to_broad = make_raw_to_broad(load_coralscapes_classes(raw_classes_path))
    cm = np.zeros((len(BROAD_CLASSES), len(BROAD_CLASSES)), dtype=np.int64)
    for row in df.itertuples():
        image = cv2.imread(row.image); raw = cv2.imread(row.mask, cv2.IMREAD_UNCHANGED); truth = broad_mask(raw, raw_to_broad)
        result = model.predict(source=image, imgsz=imgsz, conf=conf, verbose=False)[0]; pred = predicted_map(result, truth.shape)
        # A missing instance mask is a negative prediction; count it as the
        # broad "other" class so recall is not silently discarded.
        pred[pred == 255] = len(BROAD_CLASSES) - 1
        valid = truth != 255
        for t, p in zip(truth[valid].ravel(), pred[valid].ravel()):
            if t < len(BROAD_CLASSES) and p < len(BROAD_CLASSES): cm[t, p] += 1
    iou = {}
    for i, name in enumerate(BROAD_CLASSES):
        tp = cm[i, i]; denom = cm[i].sum() + cm[:, i].sum() - tp; iou[name] = float(tp / denom) if denom else 0.0
    return {"images": int(len(df)), "confidence_threshold": conf,
            "confusion_matrix": cm.tolist(), "per_class_iou": iou,
            "mIoU": float(np.mean(list(iou.values())))}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--manifest", default="data/processed/coralscapes_yolo/manifest.csv"); p.add_argument("--raw-classes", default="datasets/coralscapes/classes.json"); p.add_argument("--max-images", type=int, default=None); p.add_argument("--out", default="reports/segmentation_metrics.json")
    p.add_argument("--conf", type=float, default=0.20,
                   help="Prediction confidence threshold. Use a low value for diagnostic recall analysis.")
    args = p.parse_args(); metrics = evaluate(args.model, args.manifest, args.raw_classes, args.max_images, args.conf); Path(args.out).parent.mkdir(parents=True, exist_ok=True); Path(args.out).write_text(json.dumps(metrics, indent=2)); print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
