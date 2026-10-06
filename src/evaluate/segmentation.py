from __future__ import annotations

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO
from src.data.label_map import BROAD_CLASSES, broad_mask, load_coralscapes_classes, make_raw_to_broad
from src.trust.reliability import expected_calibration_error


def predicted_map(result, shape):
    """Render dense class map and confidence map from YOLO instance segmentation output."""
    h, w = shape
    out = np.full((h, w), 255, dtype=np.uint8)
    conf = np.full((h, w), -1.0, dtype=np.float32)
    if result.masks is None or result.boxes is None:
        return out, conf
    masks = result.masks.data.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    scores = result.boxes.conf.detach().cpu().numpy()
    for mask, cls, score in zip(masks, classes, scores):
        mask_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
        sel = mask_full & (score > conf)
        out[sel], conf[sel] = cls, score
    return out, conf


def evaluate(model_path: str | Path, manifest: str | Path, raw_classes_path: str | Path,
             max_images: int | None = None, conf: float = 0.20, imgsz: int = 512,
             out: str | Path | None = None) -> dict:
    df = pd.read_csv(manifest)
    df = df[df.split == "test"]
    if max_images:
        df = df.head(max_images)

    model = YOLO(str(model_path))
    raw_to_broad = make_raw_to_broad(load_coralscapes_classes(raw_classes_path))
    cm = np.zeros((len(BROAD_CLASSES), len(BROAD_CLASSES)), dtype=np.int64)

    all_confidences = []
    all_correctness = []

    for row in df.itertuples():
        image = cv2.imread(row.image)
        raw = cv2.imread(row.mask, cv2.IMREAD_UNCHANGED)
        truth = broad_mask(raw, raw_to_broad)

        result = model.predict(source=image, imgsz=imgsz, conf=conf, verbose=False)[0]
        pred, conf_map = predicted_map(result, truth.shape)

        # Unsegmented regions count as "other" background class for recall evaluation
        pred[pred == 255] = len(BROAD_CLASSES) - 1
        valid = truth != 255

        for t, p in zip(truth[valid].ravel(), pred[valid].ravel()):
            if t < len(BROAD_CLASSES) and p < len(BROAD_CLASSES):
                cm[t, p] += 1

        # Collect calibration data from detected instance masks
        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            scores = result.boxes.conf.detach().cpu().numpy()
            h, w = truth.shape[:2]
            for mask, cls, score in zip(masks, classes, scores):
                m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
                valid_mask = m & valid
                if valid_mask.any():
                    gt_in_mask = truth[valid_mask]
                    majority_class = int(np.bincount(gt_in_mask).argmax())
                    is_correct = int(cls == majority_class)
                    all_confidences.append(float(score))
                    all_correctness.append(is_correct)

    iou = {}
    for i, name in enumerate(BROAD_CLASSES):
        tp = cm[i, i]
        denom = cm[i].sum() + cm[:, i].sum() - tp
        iou[name] = round(float(tp / denom), 4) if denom else 0.0

    calibration_stats = expected_calibration_error(all_confidences, all_correctness, bins=10)

    metrics = {
        "images": int(len(df)),
        "confidence_threshold": conf,
        "mIoU": round(float(np.mean(list(iou.values()))), 4),
        "per_class_iou": iou,
        "calibration": calibration_stats,
        "confusion_matrix": cm.tolist(),
    }

    if out:
        out_p = Path(out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(metrics, indent=2))

    return metrics


def main():
    p = argparse.ArgumentParser(description="Evaluate YOLO segmentation model on test set")
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", default="data/processed/coralscapes_yolo/manifest.csv")
    p.add_argument("--raw-classes", default="data/processed/coralscapes_yolo/classes.json")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--conf", type=float, default=0.20,
                   help="Prediction confidence threshold.")
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--out", default="reports/segmentation_metrics.json")
    args = p.parse_args()

    metrics = evaluate(
        args.model,
        args.manifest,
        args.raw_classes,
        max_images=args.max_images,
        conf=args.conf,
        imgsz=args.imgsz,
        out=args.out,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
