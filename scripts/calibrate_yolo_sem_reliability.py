"""Calibrate YOLO semantic confidence and choose a review gate on validation data.

Uses only the official CoralscapesV2 validation Parquet split. The held-out test
fixture is never read or modified. The operating point targets at least 70%
per-image pixel accuracy while allowing at most 5% of lower-accuracy validation
images through the prediction gate; see the saved JSON for sample counts.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.label_map import (  # noqa: E402
    CLASSES,
    load_raw_labels,
    make_raw_to_project,
    project_mask,
)
from src.inference.pipeline import _confidence_semantic_predictor_class  # noqa: E402

VALIDATION_PARQUET = ROOT / "data/raw/coralscapes_v2/data/validation-00000-of-00001.parquet"
LABELS_JSON = ROOT / "data/raw/coralscapes_v2/id2label.json"
CHECKPOINT = ROOT / "models/checkpoints/yolo26n_sem/weights/best.pt"
OUTPUT = ROOT / "models/checkpoints/yolo26n_sem/reliability_calibration.json"
IMGSZ = 1024
PIXEL_ACCURACY_FLOOR = 0.70
MAX_BAD_ACCEPTANCE_RATE = 0.05
PIXEL_SAMPLE_PER_IMAGE = 2000


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """Fit one scalar temperature by minimizing sampled-pixel NLL."""
    x = torch.from_numpy(logits.astype(np.float32, copy=False))
    y = torch.from_numpy(targets.astype(np.int64, copy=False))
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=80, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = torch.nn.functional.cross_entropy(x / temperature, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).item())


def ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(confidence), 1)
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = (confidence >= lo) & (confidence <= hi if hi == 1.0 else confidence < hi)
        if selected.any():
            value += selected.sum() / total * abs(confidence[selected].mean() - correct[selected].mean())
    return float(value)


def choose_gate(records: list[dict]) -> tuple[float, float, dict]:
    """Maximize validation acceptance subject to the documented accuracy/risk targets."""
    bad = np.array([item["pixel_accuracy"] < PIXEL_ACCURACY_FLOOR for item in records])
    conf = np.array([item["mean_confidence"] for item in records])
    unc = np.array([item["mean_uncertainty"] for item in records])
    confidence_candidates = np.unique(np.r_[0.0, conf, 1.0])
    uncertainty_candidates = np.unique(np.r_[0.0, unc, 1.0])
    best = None
    bad_count = int(bad.sum())

    for min_conf in confidence_candidates:
        for max_unc in uncertainty_candidates:
            accepted = (conf >= min_conf) & (unc <= max_unc)
            false_accept_rate = float((accepted & bad).sum() / bad_count) if bad_count else 0.0
            if false_accept_rate <= MAX_BAD_ACCEPTANCE_RATE:
                accepted_good = int((accepted & ~bad).sum())
                accepted_total = int(accepted.sum())
                candidate = (accepted_good, accepted_total, -false_accept_rate, float(min_conf), float(max_unc), accepted)
                if best is None or candidate[:3] > best[:3]:
                    best = candidate

    if best is None:
        raise RuntimeError("No confidence/uncertainty thresholds met the configured bad-acceptance target")
    _, _, neg_bad_rate, min_conf, max_unc, accepted = best
    stats = {
        "validation_images": len(records),
        "images_below_accuracy_floor": bad_count,
        "accepted_images": int(accepted.sum()),
        "accepted_good_images": int((accepted & ~bad).sum()),
        "accepted_low_accuracy_images": int((accepted & bad).sum()),
        "bad_acceptance_rate": round(-neg_bad_rate, 6),
        "accepted_image_ratio": round(float(accepted.mean()), 6),
        "mean_pixel_accuracy_accepted": round(float(np.mean([r["pixel_accuracy"] for r, keep in zip(records, accepted) if keep])), 6)
        if accepted.any() else None,
    }
    return min_conf, max_unc, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    for required in (VALIDATION_PARQUET, LABELS_JSON, CHECKPOINT):
        if not required.is_file():
            raise FileNotFoundError(required)

    raw_to_project = make_raw_to_project(load_raw_labels(LABELS_JSON))
    model = YOLO(str(CHECKPOINT))
    if len(model.names) != len(CLASSES):
        raise RuntimeError(f"Expected {len(CLASSES)} YOLO classes; checkpoint has {len(model.names)}")
    predictor = _confidence_semantic_predictor_class(retain_logits=True)

    rng = np.random.default_rng(20261022)
    sampled_logits: list[np.ndarray] = []
    sampled_targets: list[np.ndarray] = []
    records: list[dict] = []
    ece_confidence: list[np.ndarray] = []
    ece_correct: list[np.ndarray] = []
    parquet = pq.ParquetFile(VALIDATION_PARQUET)
    total = parquet.metadata.num_rows

    def validation_rows():
        for batch in parquet.iter_batches(batch_size=1):
            yield from batch.to_pylist()

    for index, row in enumerate(validation_rows(), start=1):
        image = np.asarray(Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB"))[:, :, ::-1].copy()
        raw_truth = np.asarray(Image.open(io.BytesIO(row["label"]["bytes"])))
        truth = project_mask(raw_truth, raw_to_project)
        result = model.predict(
            source=image, imgsz=IMGSZ, device=args.device, verbose=False, predictor=predictor
        )[0]
        semantic = result.semantic_mask.data.detach().cpu().numpy().astype(np.int16)
        logits = result.semantic_logits.numpy().astype(np.float32)
        if logits.shape[1:] != truth.shape:
            raise RuntimeError(f"Logit/truth shape mismatch: {logits.shape[1:]} vs {truth.shape}")

        valid = (truth != 255) & (semantic >= 0) & (semantic < len(CLASSES))
        if not valid.any():
            continue
        correct = (semantic == truth) & valid
        confidence = result.semantic_confidence
        logits_flat = logits[:, valid].T
        targets_flat = truth[valid].astype(np.int64)
        sample_count = min(PIXEL_SAMPLE_PER_IMAGE, len(targets_flat))
        sample_indices = rng.choice(len(targets_flat), size=sample_count, replace=False)
        sampled_logits.append(logits_flat[sample_indices])
        sampled_targets.append(targets_flat[sample_indices])

        pixel_confidence = confidence[valid]
        pixel_correct = correct[valid]
        ece_confidence.append(pixel_confidence[sample_indices].astype(np.float32))
        ece_correct.append(pixel_correct[sample_indices].astype(np.float32))

        records.append({
            "image_index": index,
            "pixel_accuracy": float(correct.sum() / valid.sum()),
            "mean_confidence_raw": float(confidence[semantic != 255].mean()),
        })
        if index % 10 == 0 or index == total:
            print(f"Processed validation image {index}/{total}", flush=True)

    if len(records) != total:
        raise RuntimeError(f"Only scored {len(records)} of {total} validation images")

    temperature = fit_temperature(np.concatenate(sampled_logits), np.concatenate(sampled_targets))
    # Recompute image-level confidence and entropy with the fitted temperature.
    # To avoid retaining all full-resolution logits in memory, inference records
    # are re-run once at the calibrated temperature.
    model = YOLO(str(CHECKPOINT))
    predictor = _confidence_semantic_predictor_class(temperature=temperature)
    calibrated_records = []
    for index, row in enumerate(validation_rows(), start=1):
        image = np.asarray(Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB"))[:, :, ::-1].copy()
        raw_truth = np.asarray(Image.open(io.BytesIO(row["label"]["bytes"])))
        truth = project_mask(raw_truth, raw_to_project)
        result = model.predict(
            source=image, imgsz=IMGSZ, device=args.device, verbose=False, predictor=predictor
        )[0]
        semantic = result.semantic_mask.data.detach().cpu().numpy().astype(np.int16)
        valid = (truth != 255) & (semantic >= 0) & (semantic < len(CLASSES))
        calibrated_records.append({
            "image_index": index,
            "pixel_accuracy": records[index - 1]["pixel_accuracy"],
            "mean_confidence": float(result.semantic_confidence[semantic != 255].mean()),
            "mean_uncertainty": float(result.semantic_uncertainty),
        })
        if index % 10 == 0 or index == total:
            print(f"Re-scored calibrated validation image {index}/{total}", flush=True)

    min_confidence, max_uncertainty, gate_stats = choose_gate(calibrated_records)
    sample_confidence = np.concatenate(ece_confidence)
    sample_correct = np.concatenate(ece_correct)
    # Evaluate ECE after temperature scaling against the same sampled labels.
    all_logits = np.concatenate(sampled_logits)
    all_targets = np.concatenate(sampled_targets)
    probs = torch.from_numpy(all_logits / temperature).softmax(dim=1).numpy()
    calibrated_pixel_conf = probs.max(axis=1)
    calibrated_pixel_correct = (probs.argmax(axis=1) == all_targets).astype(np.float32)
    raw_ece = ece(sample_confidence, sample_correct)
    calibrated_ece = ece(calibrated_pixel_conf, calibrated_pixel_correct)

    output = {
        "model": str(CHECKPOINT.relative_to(ROOT)),
        "dataset": str(VALIDATION_PARQUET.relative_to(ROOT)),
        "split": "validation",
        "images": total,
        "class_count": len(CLASSES),
        "temperature": temperature,
        "pixel_sample_count": int(len(all_targets)),
        "pixel_ece_15_bins_before": round(raw_ece, 6),
        "pixel_ece_15_bins_after": round(calibrated_ece, 6),
        "thresholds": {
            "confidence": round(min_confidence, 6),
            "coverage": 0.05,
            "uncertainty": round(max_uncertainty, 6),
        },
        "threshold_selection": {
            "pixel_accuracy_floor": PIXEL_ACCURACY_FLOOR,
            "max_low_accuracy_acceptance_rate": MAX_BAD_ACCEPTANCE_RATE,
            **gate_stats,
        },
        "note": "Temperature and thresholds were fitted on the validation split. Validate this operating point on a separate new reef/camera before deployment.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
