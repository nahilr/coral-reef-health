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
from src.data.label_map import CLASSES, load_raw_labels, make_raw_to_project, project_mask
from src.trust.reliability import expected_calibration_error


def predicted_map(result, shape):
    """Render dense maps from YOLO semantic or instance segmentation output."""
    h, w = shape
    out = np.full((h, w), 255, dtype=np.uint8)
    conf = np.full((h, w), -1.0, dtype=np.float32)
    semantic = getattr(result, "semantic_mask", None)
    if semantic is not None:
        class_map = semantic.data.detach().cpu().numpy().astype(np.uint8)
        if class_map.shape != (h, w):
            class_map = cv2.resize(class_map, (w, h), interpolation=cv2.INTER_NEAREST)
        out[:] = class_map
        return out, conf
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
    """Evaluate a segmentation checkpoint against the 7-class project taxonomy."""
    df = pd.read_csv(manifest)
    if "split" in df.columns:
        df = df[df.split == "test"]
    if max_images:
        df = df.head(max_images)

    model = YOLO(str(model_path))
    raw_to_project = make_raw_to_project(load_raw_labels(raw_classes_path))
    # An extra column records foreground pixels missed by the detector
    cm = np.zeros((len(CLASSES), len(CLASSES) + 1), dtype=np.int64)

    all_confidences = []
    all_correctness = []

    for row in df.itertuples():
        image = cv2.imread(row.image)
        raw = cv2.imread(row.mask, cv2.IMREAD_UNCHANGED)
        truth = project_mask(raw, raw_to_project)
        predict_args = {"source": image, "imgsz": imgsz, "verbose": False}
        if getattr(model, "task", None) != "semantic":
            predict_args["conf"] = conf
        result = model.predict(**predict_args)[0]
        pred, conf_map = predicted_map(result, truth.shape)
        pred[pred == 255] = len(CLASSES)
        valid = truth != 255
        t, p = truth[valid].ravel(), pred[valid].ravel()
        np.add.at(cm, (t, p), 1)

        # Collect calibration data from instance detections if present
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

    per_class_iou = {}
    for i, name in enumerate(CLASSES):
        tp = cm[i, i]
        denom = cm[i].sum() + cm[:, i].sum() - tp
        per_class_iou[name] = round(float(tp / denom), 4) if denom else 0.0

    calibration_stats = expected_calibration_error(all_confidences, all_correctness, bins=10) if all_confidences else None

    metrics = {
        "images": int(len(df)),
        "classes": CLASSES,
        "confidence_threshold": conf,
        "mIoU": round(float(np.mean(list(per_class_iou.values()))), 4),
        "per_class_iou": per_class_iou,
        "calibration": calibration_stats,
        "confusion_matrix": cm.tolist(),
        "missed_prediction_column": len(CLASSES),
    }
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2))
    return metrics

def main():
    p = argparse.ArgumentParser(description="Evaluate segmentation model on test set")
    p.add_argument("--model", default="models/checkpoints/yolo26n_sem/weights/best.pt")
    p.add_argument("--manifest", default="data/processed/yolo/manifest.csv")
    p.add_argument("--raw-classes", default="configs/dinov3_id2label.json")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--conf", type=float, default=0.20, help="Prediction confidence threshold.")
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--out", default="reports/eval_metrics.json")
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
