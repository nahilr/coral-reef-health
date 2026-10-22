"""Compare project YOLO checkpoints and released CoralscapesV2 dense models.

Scores five backends on the same official test split and the same project
taxonomy: YOLO26n-sem, YOLO26s-sem, YOLO26n-seg, the released SegFormer-B5
(95-class) and the released DINOv3 ViT-L + LoRA + DPT (95-class).

Run with the evaluation environment and a prepared CoralscapesV2 official test
split under data/coralscapesv2-test/{images,masks}.
"""
from __future__ import annotations

import argparse
import json
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.label_map import (  # noqa: E402
    CLASSES,
    CLASS_TO_ID,
    load_raw_labels,
    make_raw_to_project,
    raw_name_to_class,
)
from src.inference.dinov3_dpt import (  # noqa: E402
    DEFAULT_REPO as DINOV3_REPO,
    dinov3_project_ids,
    load_dinov3_dpt,
    load_id2label,
    predict_project_map as predict_dinov3_map,
)

TEST_ROOT = ROOT / "data/eval/coralscapesv2-test"
YOLO_PATH = ROOT / "models/checkpoints/yolo26n_sem/weights/best.pt"
YOLO_S_PATH = ROOT / "models/checkpoints/yolo26s_sem/weights/best.pt"
YOLO_SEG_PATH = ROOT / "models/checkpoints/yolo26n_seg/weights/best.pt"
SEGFORMER_PATH = ROOT / "models/pretrained/segformer-b5-coralscapesv2-95"
DINOV3_PATH = ROOT / "models/pretrained/coralscapesv2-dinov3-vitl-lora-dpt-95_class"
DATA_LABELS = TEST_ROOT / "id2label.json"
OUT = ROOT / "models/model_comparison.json"
IMGSZ = 1024
N_CLASSES = len(CLASSES)
IGNORE = 255
PRED_IGNORE = N_CLASSES
# Cache directories are keyed by taxonomy size so a class-count change can
# never silently reuse predictions computed under a different taxonomy.
YOLO_CACHE = TEST_ROOT / f"predictions/yolo26n_sem_{N_CLASSES}cls"
YOLO_S_CACHE = TEST_ROOT / f"predictions/yolo26s_sem_{N_CLASSES}cls"
YOLO_SEG_CACHE = TEST_ROOT / f"predictions/yolo26n_seg_{N_CLASSES}cls"
SEGFORMER_CACHE = TEST_ROOT / f"predictions/segformer_b5_{N_CLASSES}cls"
DINOV3_CACHE = TEST_ROOT / f"predictions/dinov3_vitl_dpt_{N_CLASSES}cls"
MODEL_ORDER = ("yolo", "yolo_s", "yolo_seg", "segformer", "dinov3")
SEG_CONF = 0.25


def score_map(cm: np.ndarray) -> dict:
    iou = {}
    for class_id, name in enumerate(CLASSES):
        tp = int(cm[class_id, class_id])
        union = int(cm[class_id].sum() + cm[:, class_id].sum() - tp)
        iou[name] = (tp / union) if union else None
    defined = [value for value in iou.values() if value is not None]
    return {
        "pixel_accuracy": float(np.trace(cm[:, :N_CLASSES]) / max(int(cm.sum()), 1)),
        "mean_iou": float(np.mean(defined)) if defined else None,
        "classes_in_miou": len(defined),
        "per_class_iou": iou,
        "confusion_matrix": cm.tolist(),
    }


def update_cm(cm: np.ndarray, truth: np.ndarray, pred: np.ndarray) -> None:
    valid = truth != IGNORE
    t = truth[valid].astype(np.int64)
    p = pred[valid].astype(np.int64)
    p[(p < 0) | (p > PRED_IGNORE)] = PRED_IGNORE
    counts = np.bincount(t * (N_CLASSES + 1) + p, minlength=N_CLASSES * (N_CLASSES + 1))
    cm += counts.reshape(N_CLASSES, N_CLASSES + 1)


def dataset_paths() -> list[tuple[Path, Path]]:
    images = sorted([
        *(TEST_ROOT / "images").glob("test_*.jpg"),
        *(TEST_ROOT / "images").glob("test_*.png"),
    ])
    pairs = []
    for image in images:
        mask = TEST_ROOT / "masks" / f"{image.stem}.png"
        if mask.is_file():
            pairs.append((image, mask))
    if len(pairs) != 473:
        raise RuntimeError(f"Expected 473 official test image/mask pairs; found {len(pairs)}")
    return pairs


def truth_map(mask: np.ndarray, raw_to_project: dict[int, int | None]) -> np.ndarray:
    truth = np.full(mask.shape, IGNORE, dtype=np.uint8)
    for raw_id, class_id in raw_to_project.items():
        if class_id is not None:
            truth[mask == raw_id] = class_id
    return truth


def project_ids_from_labels(labels_by_id: dict[int, str], num_model_classes: int) -> list[int]:
    labels_by_id = {int(index): str(name) for index, name in labels_by_id.items()}
    ids = []
    for index in range(num_model_classes):
        name = str(labels_by_id.get(index, "")).strip()
        # Model output ID 0 is its explicit void/unlabeled category.
        if index == 0 or not name or name.lower() in {"void", "unlabeled"}:
            ids.append(PRED_IGNORE)
            continue
        project_name = raw_name_to_class(name)
        ids.append(CLASS_TO_ID[project_name] if project_name else PRED_IGNORE)
    return ids


def check_project_ids(ids: list[int], model_name: str) -> None:
    """Fail loudly if the fold would drop a project class or leak into the sink."""
    reachable = set(ids) - {PRED_IGNORE}
    missing = set(range(N_CLASSES)) - reachable
    if missing:
        names = [CLASSES[index] for index in sorted(missing)]
        raise RuntimeError(f"{model_name} label fold cannot reach project classes: {names}")


def run_yolo(
    pairs: list[tuple[Path, Path]],
    raw_to_project: dict,
    checkpoint: Path = YOLO_PATH,
    cache: Path = YOLO_CACHE,
    model_name: str = "Ultralytics YOLO26n-sem",
) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    model = None
    cm = np.zeros((N_CLASSES, N_CLASSES + 1), dtype=np.int64)
    inference_seconds = 0.0
    inference_images = 0
    for index, (image_path, mask_path) in enumerate(pairs, start=1):
        cache_path = cache / f"{image_path.stem}.png"
        if cache_path.is_file():
            pred = np.asarray(Image.open(cache_path), dtype=np.uint8)
        else:
            if model is None:
                model = YOLO(str(checkpoint))
                # Refuse to score a checkpoint trained on a different taxonomy.
                if len(model.names) != N_CLASSES:
                    raise RuntimeError(
                        f"{checkpoint} has {len(model.names)} classes but the project taxonomy has "
                        f"{N_CLASSES}; retrain the checkpoint before comparing."
                    )
            infer_start = time.perf_counter()
            result = model.predict(
                source=str(image_path), imgsz=IMGSZ, batch=1, device=0, verbose=False,
            )[0]
            inference_seconds += time.perf_counter() - infer_start
            inference_images += 1
            semantic = getattr(result, "semantic_mask", None)
            if semantic is None:
                raise RuntimeError("YOLO checkpoint did not return a semantic class map")
            pred = semantic.data.detach().cpu().numpy().astype(np.uint8)
            if pred.shape != (1024, 2048):
                pred = np.asarray(Image.fromarray(pred).resize((2048, 1024), Image.Resampling.NEAREST))
            pred[pred >= N_CLASSES] = PRED_IGNORE
            Image.fromarray(pred).save(cache_path, compress_level=1)
        raw = np.asarray(Image.open(mask_path), dtype=np.uint8)
        truth = truth_map(raw, raw_to_project)
        update_cm(cm, truth, pred)
        if index % 50 == 0:
            print(f"YOLO processed {index}/{len(pairs)} test images", flush=True)
    return {
        "model": model_name,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "images": len(pairs),
        "protocol": f"One aspect-preserving Ultralytics inference pass per 2048x1024 image, imgsz=1024 (long side 1024), FP32. Dense {N_CLASSES}-class semantic argmax; no confidence threshold.",
        "inference_images_measured": inference_images,
        "inference_seconds_measured": inference_seconds,
        "seconds_per_image": inference_seconds / inference_images if inference_images else None,
        **score_map(cm),
    }


def run_yolo_seg(pairs: list[tuple[Path, Path]], raw_to_project: dict) -> dict:
    import cv2

    YOLO_SEG_CACHE.mkdir(parents=True, exist_ok=True)
    model = None
    cm = np.zeros((N_CLASSES, N_CLASSES + 1), dtype=np.int64)
    inference_seconds = 0.0
    inference_images = 0
    uncovered_fraction = 0.0
    for index, (image_path, mask_path) in enumerate(pairs, start=1):
        cache_path = YOLO_SEG_CACHE / f"{image_path.stem}.png"
        if cache_path.is_file():
            pred = np.asarray(Image.open(cache_path), dtype=np.uint8)
        else:
            if model is None:
                model = YOLO(str(YOLO_SEG_PATH))
                if len(model.names) != N_CLASSES:
                    raise RuntimeError(
                        f"{YOLO_SEG_PATH} has {len(model.names)} classes but the project taxonomy has "
                        f"{N_CLASSES}; retrain the checkpoint before comparing."
                    )
            infer_start = time.perf_counter()
            result = model.predict(
                source=str(image_path), imgsz=IMGSZ, conf=SEG_CONF, batch=1, device=0, verbose=False,
            )[0]
            inference_seconds += time.perf_counter() - infer_start
            inference_images += 1
            height, width = result.orig_shape[:2]
            # Instance masks are rasterised to a semantic map: the highest-scoring
            # detection wins each pixel. Pixels no mask covers stay unlabelled and
            # count as missed predictions, matching the strict dense scoring used
            # for the semantic models.
            class_map = np.full((height, width), PRED_IGNORE, dtype=np.uint8)
            best_score = np.zeros((height, width), dtype=np.float32)
            if result.masks is not None and result.boxes is not None:
                masks = result.masks.data.detach().cpu().numpy()
                classes = result.boxes.cls.detach().cpu().numpy().astype(int)
                scores = result.boxes.conf.detach().cpu().numpy()
                for mask, class_id, score in zip(masks, classes, scores):
                    filled = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 0.5
                    take = filled & (score >= best_score)
                    best_score[take] = score
                    class_map[take] = class_id
            uncovered_fraction += float((class_map == PRED_IGNORE).mean())
            pred = np.array(
                Image.fromarray(class_map).resize((2048, 1024), Image.Resampling.NEAREST),
                dtype=np.uint8,
            )
            pred[pred >= N_CLASSES] = PRED_IGNORE
            Image.fromarray(pred).save(cache_path, compress_level=1)
        raw = np.asarray(Image.open(mask_path), dtype=np.uint8)
        truth = truth_map(raw, raw_to_project)
        update_cm(cm, truth, pred)
        if index % 50 == 0:
            print(f"YOLO-seg processed {index}/{len(pairs)} test images", flush=True)
    return {
        "model": "Ultralytics YOLO26n-seg",
        "checkpoint": str(YOLO_SEG_PATH.relative_to(ROOT)),
        "images": len(pairs),
        "protocol": f"One aspect-preserving Ultralytics inference pass per 2048x1024 image, imgsz=1024, conf={SEG_CONF}, FP32. Instance masks rasterised to a dense map with highest-confidence-wins; uncovered pixels count as missed predictions.",
        "confidence_threshold": SEG_CONF,
        "uncovered_pixel_pct": 100.0 * uncovered_fraction / len(pairs),
        "inference_images_measured": inference_images,
        "inference_seconds_measured": inference_seconds,
        "seconds_per_image": inference_seconds / inference_images if inference_images else None,
        **score_map(cm),
    }


def run_segformer(pairs: list[tuple[Path, Path]], raw_to_project: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    processor = SegformerImageProcessor.from_pretrained(str(SEGFORMER_PATH))
    model = SegformerForSemanticSegmentation.from_pretrained(str(SEGFORMER_PATH)).to(device).eval()
    SEGFORMER_CACHE.mkdir(parents=True, exist_ok=True)
    model_map = project_ids_from_labels(model.config.id2label, model.config.num_labels)
    if len(model_map) != model.config.num_labels or model.config.num_labels != 96:
        raise RuntimeError(f"Unexpected SegFormer label count: {model.config.num_labels}")
    check_project_ids(model_map, "SegFormer")
    # Check that the model taxonomy and ground-truth taxonomy use the same
    # spelling for every shared CoralscapesV2 class name.
    model_names = {str(v).strip().lower() for k, v in model.config.id2label.items() if str(k) != "0"}
    data_names = {name.strip().lower() for name in load_raw_labels(DATA_LABELS).values() if name.strip().lower() != "unlabeled"}
    if model_names != data_names:
        raise RuntimeError("SegFormer and dataset id2label names differ; refusing ID-based comparison")

    grouped_ids = torch.tensor(model_map, dtype=torch.long, device=device)
    cm = np.zeros((N_CLASSES, N_CLASSES + 1), dtype=np.int64)
    inference_seconds = 0.0
    inference_images = 0
    for index, (image_path, mask_path) in enumerate(pairs, start=1):
        raw = np.asarray(Image.open(mask_path), dtype=np.uint8)
        truth = truth_map(raw, raw_to_project)
        cache_path = SEGFORMER_CACHE / f"{image_path.stem}.png"
        if cache_path.is_file():
            pred = np.asarray(Image.open(cache_path), dtype=np.uint8)
        else:
            infer_start = time.perf_counter()
            # Use the model card's published single-pass protocol (768x1376).
            # This is a documented alternative to its slower two-window mode.
            image = Image.open(image_path).convert("RGB").resize((1376, 768), Image.Resampling.BILINEAR)
            inputs = processor(image, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                # FP32 only: CUDA FP16 autocast yields NaN logits for this
                # checkpoint on the local card and collapses every pixel to
                # class 0 (see src/inference/pipeline.py).
                logits = model(**inputs).logits
                logits = F.interpolate(logits.float(), size=(768, 1376), mode="bilinear", align_corners=False)
                probabilities = logits.softmax(dim=1)
                grouped = torch.zeros(
                    (1, N_CLASSES + 1, 768, 1376), dtype=probabilities.dtype, device=device
                )
                grouped.scatter_add_(
                    1,
                    grouped_ids.view(1, -1, 1, 1).expand(1, -1, 768, 1376),
                    probabilities,
                )
                pred_small = grouped.argmax(dim=1)[0].to("cpu", dtype=torch.uint8).numpy()
            pred = np.asarray(Image.fromarray(pred_small).resize((2048, 1024), Image.Resampling.NEAREST))
            Image.fromarray(pred).save(cache_path, compress_level=1)
            inference_seconds += time.perf_counter() - infer_start
            inference_images += 1
        update_cm(cm, truth, pred)
        if index % 25 == 0:
            print(f"SegFormer processed {index}/{len(pairs)} test images", flush=True)
    return {
        "model": "SegFormer-B5 fine-tuned on CoralscapesV2 (95 classes)",
        "model_id": "EPFL-ECEO/segformer-b5-finetuned-coralscapesv2-1024-1024-95_class",
        "checkpoint": str(SEGFORMER_PATH.relative_to(ROOT)),
        "device": str(device),
        "images": len(pairs),
        "protocol": "One model-card single-pass inference at 768x1376 per 2048x1024 image (direct resize), then nearest-neighbor resize of the project-class map to source dimensions. Sum softmax probability across fine labels mapped by config id2label names into each project class (void/ignored labels grouped as missed prediction).",
        "inference_images_measured": inference_images,
        "inference_seconds_measured": inference_seconds,
        "seconds_per_image": inference_seconds / inference_images if inference_images else None,
        **score_map(cm),
    }


def run_dinov3(pairs: list[tuple[Path, Path]], raw_to_project: dict,
               device_name: str | None = None) -> dict:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model, processor = load_dinov3_dpt(DINOV3_PATH, device=str(device))
    labels = load_id2label(DINOV3_PATH)
    dataset_labels = load_raw_labels(DATA_LABELS)
    normalised = {index: name.strip().lower() for index, name in labels.items()}
    if normalised != {index: name.strip().lower() for index, name in dataset_labels.items()}:
        raise RuntimeError("DINOv3 and dataset id2label names differ; refusing ID-based comparison")
    num_model_classes = int(json.loads((DINOV3_PATH / "config.json").read_text())["num_classes"])
    model_map = dinov3_project_ids(labels, num_model_classes)
    if num_model_classes != 96:
        raise RuntimeError(f"Unexpected DINOv3+DPT label count: {num_model_classes}")
    check_project_ids(model_map, "DINOv3+DPT")
    DINOV3_CACHE.mkdir(parents=True, exist_ok=True)
    cm = np.zeros((N_CLASSES, N_CLASSES + 1), dtype=np.int64)
    inference_seconds = 0.0
    inference_images = 0
    for index, (image_path, mask_path) in enumerate(pairs, start=1):
        cache_path = DINOV3_CACHE / f"{image_path.stem}.png"
        if cache_path.is_file():
            pred = np.asarray(Image.open(cache_path), dtype=np.uint8)
        else:
            infer_start = time.perf_counter()
            image = Image.open(image_path).convert("RGB")
            pred_small = predict_dinov3_map(model, processor, image, model_map, device=str(device))
            pred = np.asarray(
                Image.fromarray(pred_small).resize((2048, 1024), Image.Resampling.NEAREST),
                dtype=np.uint8,
            )
            Image.fromarray(pred).save(cache_path, compress_level=1)
            inference_seconds += time.perf_counter() - infer_start
            inference_images += 1
        raw = np.asarray(Image.open(mask_path), dtype=np.uint8)
        truth = truth_map(raw, raw_to_project)
        update_cm(cm, truth, pred)
        if index % 25 == 0:
            print(f"DINOv3+DPT processed {index}/{len(pairs)} test images", flush=True)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "model": "DINOv3 ViT-L + LoRA + DPT fine-tuned on CoralscapesV2 (95 classes)",
        "model_id": DINOV3_REPO,
        "checkpoint": str(DINOV3_PATH.relative_to(ROOT)),
        "device": str(device),
        "images": len(pairs),
        "protocol": "One model-card single-pass inference at 768x1376 per 2048x1024 image (direct resize), then nearest-neighbor resize of the project-class map to source dimensions. Sum softmax probability across fine labels mapped by dataset id2label names into each project class (void/ID 0 and unmapped labels grouped as missed prediction).",
        "inference_images_measured": inference_images,
        "inference_seconds_measured": inference_seconds,
        "seconds_per_image": inference_seconds / inference_images if inference_images else None,
        **score_map(cm),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default=",".join(MODEL_ORDER),
        help=f"Comma-separated subset of {','.join(MODEL_ORDER)} (default: all)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for the dense backends, e.g. cuda or cpu (default: auto)",
    )
    args = parser.parse_args()
    wanted = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = set(wanted) - set(MODEL_ORDER)
    if unknown or not wanted:
        raise SystemExit(f"Unknown or empty --models value; choose from {', '.join(MODEL_ORDER)}")

    if "yolo" in wanted and not YOLO_PATH.is_file():
        raise FileNotFoundError(f"YOLO checkpoint is missing: {YOLO_PATH}")
    if "yolo_s" in wanted and not YOLO_S_PATH.is_file():
        raise FileNotFoundError(f"YOLO26s-sem checkpoint is missing: {YOLO_S_PATH}")
    if "yolo_seg" in wanted and not YOLO_SEG_PATH.is_file():
        raise FileNotFoundError(f"YOLO-seg checkpoint is missing: {YOLO_SEG_PATH}")
    if "segformer" in wanted and not (SEGFORMER_PATH / "model.safetensors").is_file():
        raise FileNotFoundError(f"SegFormer checkpoint is missing: {SEGFORMER_PATH}")
    if "dinov3" in wanted and not (DINOV3_PATH / "model.safetensors").is_file():
        raise FileNotFoundError(
            f"DINOv3+DPT checkpoint is missing: {DINOV3_PATH}. Download {DINOV3_REPO} there first."
        )

    pairs = dataset_paths()
    labels = load_raw_labels(DATA_LABELS)
    raw_to_project = make_raw_to_project(labels)
    runners = {
        "yolo": lambda: run_yolo(pairs, raw_to_project),
        "yolo_s": lambda: run_yolo(
            pairs, raw_to_project, YOLO_S_PATH, YOLO_S_CACHE, "Ultralytics YOLO26s-sem"
        ),
        "yolo_seg": lambda: run_yolo_seg(pairs, raw_to_project),
        "segformer": lambda: run_segformer(pairs, raw_to_project),
        "dinov3": lambda: run_dinov3(pairs, raw_to_project, device_name=args.device),
    }
    models = [runners[name]() for name in MODEL_ORDER if name in wanted]
    result = {
        "dataset": "josauder/coralscapesV2",
        "split": "official test",
        "images": len(pairs),
        "image_shape": [1024, 2048],
        "project_classes": CLASSES,
        "ground_truth_mapping": "Raw CoralscapesV2 ids are mapped by class name using src/data/label_map.py and configs/labels.yaml; unmapped GT pixels (255) are excluded.",
        "metric_definition": "Pixel accuracy and macro IoU over valid GT pixels; predictions falling in void/ignored categories count as incorrect. Macro IoU averages classes with nonzero union.",
        "models": models,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps({"report": str(OUT), "summary": [
        {"model": m["model"], "pixel_accuracy": m["pixel_accuracy"], "mean_iou": m["mean_iou"], "seconds_per_image": m["seconds_per_image"]}
        for m in result["models"]
    ]}, indent=2))


if __name__ == "__main__":
    main()
