from __future__ import annotations

import inspect
import json
import shutil
import subprocess
from pathlib import Path
from typing import Protocol, overload
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.data.label_map import CLASSES, CLASS_TO_ID, raw_name_to_class, EXCLUDED_FROM_HEALTH
from src.data.quality import assess_image
from src.trust.reliability import prediction_uncertainty, review_decision


class VideoProgressCallback(Protocol):
    """Callback accepting either legacy progress or live frame data."""

    @overload
    def __call__(self, current: int, total: int) -> None: ...

    @overload
    def __call__(
        self,
        current: int,
        total: int,
        overlay_bgr: np.ndarray | None,
        frame_report: dict,
    ) -> None: ...


# BGR colours for the 7-class taxonomy (configs/labels.yaml).
CLASS_COLORS = np.array([
    [70, 170, 70],    # 0 coral_alive (green)
    [255, 230, 80],   # 1 coral_bleached (white/yellow)
    [150, 80, 80],    # 2 coral_dead (dark red/brown)
    [0, 160, 120],    # 3 algae (moss/olive green)
    [140, 130, 120],  # 4 hard_substrate (slate stone/rubble)
    [194, 178, 128],  # 5 sand (sand beige)
    [30, 30, 30],     # 6 non_reef (near-black discard sink)
], dtype=np.uint8)

# Scene classes that are predicted but removed from the reef health denominator.
_HEALTH_EXCLUDED = set(EXCLUDED_FROM_HEALTH)

# Live-cover tiers, best to worst (Reef Check / Gomez et al. benchmark).
_TIER_ORDER = ("Excellent", "Good", "Fair", "Poor / Degraded")

# Coral-condition disturbances cap the cover tier: a bleached or dead
# assemblage must not be reported as healthier than its condition allows.
# Level 1 (>=10% of coral affected) caps at Good, level 2 (>=25%) at Fair and
# level 3 (>=50%) at Poor / Degraded.
_CONDITION_CAPS = {0: None, 1: "Good", 2: "Fair", 3: "Poor / Degraded"}


def _confidence_semantic_predictor_class(temperature: float = 1.0, retain_logits: bool = False):
    """Build an Ultralytics predictor that retains dense semantic scores.

    Ultralytics' public ``semantic_mask`` is an argmax class map. Its semantic
    predictor receives logits before that argmax, so retain per-pixel
    probabilities there while preserving Ultralytics' own resize/letterbox
    handling for the class map.
    """
    from ultralytics.models.yolo.semantic.predict import SemanticSegmentationPredictor
    from ultralytics.utils import ops

    class ConfidenceSemanticPredictor(SemanticSegmentationPredictor):
        def postprocess(self, preds, img, orig_imgs):
            raw = preds[0] if isinstance(preds, (tuple, list)) else preds
            results = super().postprocess(preds, img, orig_imgs)

            # Exported graphs may bake argmax into the model, leaving only a
            # [B,H,W] class map. Confidence cannot be recovered in that case.
            if not isinstance(raw, torch.Tensor) or raw.ndim != 4:
                for result in results:
                    result.semantic_confidence = None
                    result.semantic_uncertainty = None
                return results

            for index, result in enumerate(results):
                logits = raw[index : index + 1].float()
                if logits.shape[2:] != img.shape[2:]:
                    logits = torch.nn.functional.interpolate(
                        logits, size=img.shape[2:], mode="bilinear", align_corners=False
                    )
                logits = ops.scale_masks(logits, result.orig_shape)[0]

                if logits.shape[0] == 1:
                    scaled_logits = logits[0] / max(float(temperature), 1e-6)
                    positive = torch.sigmoid(scaled_logits)
                    confidence = torch.where(scaled_logits >= 0, positive, 1.0 - positive)
                    probabilities = torch.cat((1.0 - positive[None], positive[None]), dim=0)
                else:
                    probabilities = (logits / max(float(temperature), 1e-6)).softmax(dim=0)
                    confidence = probabilities.max(dim=0).values

                class_map = result.semantic_mask.data
                valid = class_map != 255
                if bool(valid.any()):
                    entropy = -(probabilities.clamp_min(1e-8) * probabilities.clamp_min(1e-8).log()).sum(dim=0)
                    entropy = entropy / np.log(probabilities.shape[0]) if probabilities.shape[0] > 1 else entropy
                    result.semantic_uncertainty = float(entropy[valid].mean().clamp(0.0, 1.0).item())
                else:
                    result.semantic_uncertainty = 1.0
                result.semantic_confidence = confidence.detach().cpu().numpy().astype(np.float32)
                result.semantic_logits = logits.detach().cpu() if retain_logits else None

            return results

    return ConfidenceSemanticPredictor


def _worse_tier(a: str, b: str) -> str:
    return _TIER_ORDER[max(_TIER_ORDER.index(a), _TIER_ORDER.index(b))]


def _condition_severity(value: float, moderate: str, severe: str, extensive: str) -> tuple[int, str | None]:
    """Grade a disturbance ratio (10% / 25% / 50%) and name it."""
    if value >= 50.0:
        return 3, extensive
    if value >= 25.0:
        return 2, severe
    if value >= 10.0:
        return 1, moderate
    return 0, None


def _reef_health_verdict(lcc: float, bleaching_ratio: float, mortality_ratio: float,
                         algae_cover: float, has_reef: bool) -> tuple[str, str, str | None, str | None]:
    """Combine the live-cover tier with coral condition.

    Returns ``(category, cover_tier, condition, condition_cap)``; ``condition`` is
    ``None`` only when no reef area was detected.
    """
    if not has_reef:
        indeterminate = "Indeterminate (No reef area detected)"
        return indeterminate, indeterminate, None, None

    if lcc >= 50.0:
        cover_tier = "Excellent"
    elif lcc >= 25.0:
        cover_tier = "Good"
    elif lcc >= 10.0:
        cover_tier = "Fair"
    else:
        cover_tier = "Poor / Degraded"

    bleach_level, bleach_label = _condition_severity(
        bleaching_ratio, "Moderate Bleaching Warning", "Severe Bleaching Alert", "Extensive Bleaching"
    )
    mortality_level, mortality_label = _condition_severity(
        mortality_ratio, "Elevated Mortality", "High Mortality", "Extensive Mortality"
    )
    condition_cap = _CONDITION_CAPS[max(bleach_level, mortality_level)]
    category = cover_tier if condition_cap is None else _worse_tier(cover_tier, condition_cap)

    reasons = [label for level, label in ((bleach_level, bleach_label), (mortality_level, mortality_label)) if level > 0]
    condition = " + ".join(reasons) if reasons else "Stable"
    if algae_cover > 30.0 and algae_cover > lcc:
        reasons.append("Algae-Covered Area Warning")
    if reasons:
        category += f" [{'; '.join(reasons)}]"
    return category, cover_tier, condition, condition_cap


def compute_health_indicators(counts: dict[str, int], total_valid: int) -> dict:
    """Compute quantitative reef health metrics from segmented benthic pixels.

    Ratios use the *reef* area as the denominator: scene classes excluded from
    the reef assessment (``non_reef`` and discard sinks) are subtracted from
    ``total_valid`` first, so water, fish, or transect hardware cannot inflate
    or deflate coral cover.

    ``reef_health_category`` combines two axes: the live-coral-cover tier
    (Reef Check / Gomez) and the coral-condition severity (bleaching and
    mortality as a share of all coral). Condition can only *lower* the tier,
    never raise it. The cover-only tier remains in ``live_cover_tier``.
    """
    norm = {str(name).lower(): int(value) for name, value in counts.items()}

    def pct(value: int, denominator: int) -> float:
        return round(value / denominator * 100.0, 2) if denominator > 0 else 0.0

    live = norm.get("coral_alive", 0)
    bleached = norm.get("coral_bleached", 0)
    dead = norm.get("coral_dead", 0)
    algae = norm.get("algae", 0)
    hard_substrate = norm.get("hard_substrate", 0)
    sand = norm.get("sand", 0)
    non_reef = norm.get("non_reef", 0)

    excluded = sum(norm.get(name, 0) for name in _HEALTH_EXCLUDED)
    reef_total = int(total_valid) - excluded

    total_coral = live + bleached + dead
    bleaching_ratio = pct(bleached, total_coral)
    mortality_ratio = pct(dead, total_coral)
    lcc = pct(live, reef_total)
    algae_cover = pct(algae, reef_total)

    cat, cover_tier, condition, condition_cap = _reef_health_verdict(
        lcc, bleaching_ratio, mortality_ratio, algae_cover, reef_total > 0
    )

    return {
        "live_coral_cover_pct": lcc,
        "bleached_coral_cover_pct": pct(bleached, reef_total),
        "coral_dead_cover_pct": pct(dead, reef_total),
        "algae_covered_cover_pct": algae_cover,
        "macroalgae_cover_pct": algae_cover,
        "hard_substrate_cover_pct": pct(hard_substrate, reef_total),
        "sand_cover_pct": pct(sand, reef_total),
        "total_coral_cover_pct": pct(total_coral, reef_total),
        "bleaching_ratio_pct": bleaching_ratio,
        "mortality_ratio_pct": mortality_ratio,
        "coral_to_algae_ratio": round(live / algae, 2) if algae > 0 else 0.0,
        "non_reef_cover_pct": pct(non_reef, total_valid),
        "reef_area_pixels": max(reef_total, 0),
        "excluded_area_pixels": excluded,
        "reef_health_category": cat,
        "live_cover_tier": cover_tier,
        "coral_condition": condition,
        "condition_cap": condition_cap,
    }


# Dense backends answer "which class is this pixel", not "which colony is this".
# Boxes are drawn around connected coral areas.
BOX_CLASSES = ("coral_alive", "coral_bleached", "coral_dead")


def detections_from_class_map(
    class_map: np.ndarray,
    confidence_map: np.ndarray | None,
    class_names: list[str],
    max_detections: int = 15,
    min_area_frac: float = 0.001,
    min_fill_ratio: float = 0.2,
    max_box_frac: float = 0.35,
) -> list[dict]:
    """Turn a dense class map into drawable region boxes for coral classes.

    Each connected component of a coral class becomes one detection, keeping the
    largest ``max_detections``.
    """
    height, width = class_map.shape
    min_area = max(64, int(round(height * width * min_area_frac)))
    found: list[dict] = []

    for class_id, name in enumerate(class_names):
        if name not in BOX_CLASSES:
            continue
        binary = (class_map == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            x, y, box_w, box_h = cv2.boundingRect(contour)
            if area / float(max(1, box_w * box_h)) < min_fill_ratio:
                continue
            if (box_w * box_h) / float(height * width) > max_box_frac:
                continue
            score = None
            if confidence_map is not None:
                roi = class_map[y:y + box_h, x:x + box_w] == class_id
                score = float(confidence_map[y:y + box_h, x:x + box_w][roi].mean()) if roi.any() else None
            found.append({
                "class": name,
                "confidence": score,
                "pixels": int(area),
                "bbox": [int(x), int(y), int(x + box_w), int(y + box_h)],
            })

    found.sort(key=lambda det: (det["pixels"], det["confidence"] or 0.0), reverse=True)
    del found[max_detections:]
    for index, det in enumerate(found):
        det["id"] = index
    return found


class CoralPipeline:
    def __init__(self, model_path: str | Path = "models/checkpoints/yolo26n_sem/weights/best.pt",
                 imgsz: int = 512, conf: float = 0.20, thresholds: dict | None = None,
                 model_type: str = "yolo"):
        self.model_path = str(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.thresholds = thresholds
        self.model_type = model_type
        self.semantic_temperature = 1.0
        self.model = None
        self.class_names = CLASSES
        self.colors = CLASS_COLORS

        if thresholds is None and model_type in {"segformer", "dinov3"}:
            self.thresholds = {"confidence": 0.70, "uncertainty": 0.40}

        if model_type == "segformer":
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_path).to(self.device).eval()
            self.processor = SegformerImageProcessor.from_pretrained(self.model_path)
            self.model_task = "semantic"
            self._dense_project_ids = []
            for index in range(self.model.config.num_labels):
                name = str(self.model.config.id2label[index]).strip()
                project_class = None if index == 0 or name.lower() in {"void", "unlabeled"} else raw_name_to_class(name)
                self._dense_project_ids.append(CLASS_TO_ID[project_class] if project_class else len(CLASSES))
        elif model_type == "dinov3":
            from src.inference.dinov3_dpt import dinov3_project_ids, load_dinov3_dpt, load_id2label

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model, self.processor = load_dinov3_dpt(self.model_path, device=str(self.device))
            self.model_task = "semantic"
            num_labels = int(json.loads((Path(self.model_path) / "config.json").read_text())["num_classes"])
            self._dense_project_ids = dinov3_project_ids(load_id2label(self.model_path), num_labels)
        elif model_type == "yolo":
            self.model = YOLO(self.model_path)
            self.model_task = getattr(self.model, "task", "segment")
            model_names = self.model.names
            self.class_names = [model_names[i] for i in range(len(model_names))] if isinstance(model_names, dict) else list(model_names)
            if self.model_task == "semantic":
                calibration_path = Path(self.model_path).parent.parent / "reliability_calibration.json"
                if calibration_path.is_file():
                    calibration = json.loads(calibration_path.read_text())
                    self.semantic_temperature = float(calibration.get("temperature", 1.0))
                    if thresholds is None:
                        self.thresholds = calibration.get("thresholds")
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    @property
    def model_display_name(self) -> str:
        if self.model_type == "dinov3":
            return "DINOv3-DPT"
        if self.model_type == "segformer":
            return "SegFormer-B5"
        model_yaml = getattr(getattr(self.model, "model", None), "yaml", {}) or {}
        yaml_file = model_yaml.get("yaml_file")
        if yaml_file:
            model_name = Path(yaml_file).stem
            if model_name.lower().startswith("yolo"):
                return "YOLO" + model_name[4:]
        norm = self.model_path.lower()
        if "yolo26s_sem" in norm:
            return "YOLO26s-sem"
        if "yolo26n_sem" in norm:
            return "YOLO26n-sem"
        if "yolo26n_seg" in norm:
            return "YOLO26n-seg"
        p = Path(self.model_path)
        return p.parent.parent.name if p.name == "best.pt" else p.stem

    def infer(self, image_or_path: str | Path | np.ndarray,
              save_overlay: str | Path | None = None,
              return_overlay: bool = False) -> dict:
        if isinstance(image_or_path, (str, Path)):
            image = cv2.imread(str(image_or_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot read image: {image_or_path}")
            source_desc = str(image_or_path)
        elif isinstance(image_or_path, np.ndarray):
            image = image_or_path
            source_desc = "in_memory_frame"
        else:
            raise ValueError(f"Unsupported image input type: {type(image_or_path)}")

        quality = assess_image(image)
        semantic_confidence = None
        dense_uncertainty = None
        if self.model_type in {"segformer", "dinov3"}:
            semantic_map, semantic_confidence, dense_uncertainty = (
                self._predict_segformer(image) if self.model_type == "segformer"
                else self._predict_dinov3(image)
            )
            semantic_output = semantic_map
        else:
            predict_args = {"source": image, "imgsz": self.imgsz, "conf": self.conf, "verbose": False}
            if self.model_task == "semantic":
                predict_args["predictor"] = _confidence_semantic_predictor_class(
                    temperature=self.semantic_temperature
                )
            result = self.model.predict(**predict_args)[0]
            semantic_output = getattr(result, "semantic_mask", None)
            if semantic_output is not None:
                semantic_confidence = getattr(result, "semantic_confidence", None)
                dense_uncertainty = getattr(result, "semantic_uncertainty", None)

        height, width = image.shape[:2]
        class_map = np.full((height, width), -1, dtype=np.int16)
        confidence_map = np.zeros((height, width), dtype=np.float32)
        detections = []

        if semantic_output is not None:
            if isinstance(semantic_output, np.ndarray):
                semantic_map = semantic_output.astype(np.int16, copy=False)
            else:
                semantic_map = semantic_output.data.detach().cpu().numpy().astype(np.int16)
            if semantic_map.shape != (height, width):
                semantic_map = cv2.resize(semantic_map, (width, height), interpolation=cv2.INTER_NEAREST)
            recognized = (semantic_map >= 0) & (semantic_map < len(self.class_names))
            class_map[recognized] = semantic_map[recognized]
            if semantic_confidence is not None:
                if semantic_confidence.shape != (height, width):
                    semantic_confidence = cv2.resize(
                        semantic_confidence, (width, height), interpolation=cv2.INTER_LINEAR
                    )
                confidence_map[recognized] = semantic_confidence[recognized].astype(np.float32)
        elif result.masks is not None and result.boxes is not None:
            masks = result.masks.data.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            scores = result.boxes.conf.detach().cpu().numpy()
            boxes = result.boxes.xyxy.detach().cpu().numpy()

            for idx, (mask, cls, score, (x1, y1, x2, y2)) in enumerate(zip(masks, classes, scores, boxes)):
                mask_full = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 0.5
                replace = mask_full & (score >= confidence_map)
                class_map[replace] = cls
                confidence_map[replace] = score

                cls_name = self.class_names[cls] if cls < len(self.class_names) else str(cls)
                det_info = {
                    "id": idx,
                    "class": cls_name,
                    "confidence": float(score),
                    "pixels": int(mask_full.sum()),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
                detections.append(det_info)

        # Dense backends emit a class map: derive region boxes for coral classes
        if semantic_output is not None:
            detections = detections_from_class_map(
                class_map,
                confidence_map if semantic_confidence is not None else None,
                self.class_names,
            )

        valid = class_map >= 0
        total_valid = int(valid.sum())
        counts = {name: int((class_map == i).sum()) for i, name in enumerate(self.class_names)}
        scores_available = semantic_confidence is not None or (semantic_output is None and bool(detections))
        mean_conf = float(confidence_map[valid].mean()) if (valid.any() and scores_available) else None
        excluded = sum(counts.get(name, 0) for name in _HEALTH_EXCLUDED)
        coverage = max(0, total_valid - excluded) / float(width * height)

        if dense_uncertainty is not None:
            uncertainty = dense_uncertainty
        elif semantic_output is not None:
            uncertainty = None
        else:
            uncertainty = prediction_uncertainty([d["confidence"] for d in detections])

        # Health indicators
        proportions = {name: round(counts[name] / total_valid, 4) if total_valid else 0.0 for name in self.class_names}
        health = compute_health_indicators(counts, total_valid)

        # Trust / Reliability gate
        decision = review_decision(quality, mean_conf, coverage, uncertainty, thresholds=self.thresholds)

        if save_overlay or return_overlay:
            overlay = image.copy()
            for cls_idx, color in enumerate(self.colors):
                sel = class_map == cls_idx
                if sel.any():
                    overlay[sel] = (0.55 * overlay[sel] + 0.45 * color).astype(np.uint8)

            small_banner = width < 640
            fs = max(0.32, min(0.95, width / 1400.0))
            bt = max(1, int(round(width / 750)))
            row_h = int(22 * fs) + 6
            bar_h = max(6, int(10 * fs))
            if small_banner:
                banner_h = min(height, int(row_h * 4 + bar_h + 16))
            else:
                banner_h = min(int(height * 0.32), int(row_h * 2.9 + bar_h + 16))
            banner_y = height - banner_h

            label_rects: list[tuple[int, int, int, int]] = [(0, banner_y, width, height)]

            def overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
                return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

            pending_labels: list[tuple[str, tuple[int, int, int], tuple[int, int, int, int]]] = []
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cls_name = det["class"]
                cls_idx = self.class_names.index(cls_name) if cls_name in self.class_names else 0
                color = tuple(int(c) for c in self.colors[cls_idx])

                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, bt)
                label = cls_name if det["confidence"] is None else f"{cls_name} {det['confidence']:.2f}"
                pending_labels.append((label, color, (x1, y1, x2, y2)))

            label_fs = fs * 0.7 if semantic_output is not None else fs
            for label, color, (x1, y1, x2, y2) in pending_labels:
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_fs, bt)
                label_x = max(0, min(x1, width - tw - 8))
                candidates = [(label_x, y1 - th - 6), (label_x, y1 + 2), (label_x, y2 - th - 2)]
                for lx, ly in candidates:
                    if ly < 0 or ly + th + 6 > height:
                        continue
                    rect = (lx, ly, lx + tw + 6, ly + th + 6)
                    if any(overlaps(rect, other) for other in label_rects):
                        continue
                    label_rects.append(rect)
                    cv2.rectangle(overlay, (lx, ly), (lx + tw + 6, ly + th + 6), color, -1)
                    cv2.putText(overlay, label, (lx + 3, ly + th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                                label_fs, (0, 0, 0), bt, cv2.LINE_AA)
                    break

            # Bottom overlay banner
            banner_roi = overlay[banner_y:height, 0:width]
            bg = np.zeros_like(banner_roi)
            cv2.addWeighted(bg, 0.78, banner_roi, 0.22, 0, banner_roi)

            # Subtle neutral top border line
            cv2.line(overlay, (0, banner_y), (width, banner_y), (75, 75, 75), max(1, int(1.5 * fs)))

            # Health verdict accent color
            tier_text, _, reasons_text = health["reef_health_category"].partition(" [")
            tier_lower = tier_text.lower()
            if "excellent" in tier_lower or "good" in tier_lower:
                health_accent = (60, 200, 80)      # Emerald green
            elif "fair" in tier_lower:
                health_accent = (40, 180, 240)     # Golden amber
            elif "poor" in tier_lower or "degraded" in tier_lower:
                health_accent = (50, 60, 230)      # Coral red
            else:
                health_accent = (160, 160, 160)    # Slate

            # Trust status accent color
            trust_accent = (60, 200, 80) if decision["status"] == "accepted" else (50, 60, 230)

            margin = max(6 if small_banner else 12, int(width * 0.015))
            bar_w = width - (2 * margin)

            # Row 1: Health Verdict & Telemetry
            y1 = banner_y + int(row_h * 0.95)
            cond_str = f" [{reasons_text.rstrip(']')}]" if reasons_text else ""
            prefix = "REEF HEALTH: "
            verdict = f"{tier_text}{cond_str}"
            metrics = f"  |  LCC: {health['live_coral_cover_pct']}%  |  Bleached: {health['bleached_coral_cover_pct']}%  |  Algae: {health['algae_covered_cover_pct']}%"

            conf_str = f"{mean_conf:.2f}" if mean_conf is not None else "n/a"
            status_str = decision["status"].upper()
            r_model = f"Model: {self.model_display_name}  |  "
            r_trust = f"TRUST: {status_str}"
            r_stats = f"  |  Cov: {coverage*100:.1f}%  |  Conf: {conf_str}"

            if small_banner:
                def draw_fitted_line(text: str, baseline: int, color: tuple[int, int, int], scale: float) -> None:
                    while scale > 0.16:
                        text_w = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
                        if text_w <= bar_w:
                            break
                        scale -= 0.02
                    cv2.putText(
                        overlay, text, (margin, baseline), cv2.FONT_HERSHEY_SIMPLEX,
                        max(scale, 0.16), color, 1, cv2.LINE_AA,
                    )

                small_scale = min(0.36, fs * 0.82)
                draw_fitted_line(f"REEF HEALTH: {tier_text}", y1, health_accent, small_scale)
                draw_fitted_line(
                    f"LCC: {health['live_coral_cover_pct']}% | Bleached: {health['bleached_coral_cover_pct']}% | Algae: {health['algae_covered_cover_pct']}%",
                    y1 + row_h, (235, 235, 235), small_scale,
                )
                draw_fitted_line(
                    f"{self.model_display_name} | {status_str}",
                    y1 + (2 * row_h), trust_accent, small_scale,
                )
                bar_y = y1 + (2 * row_h) + max(4, int(row_h * 0.45))
            else:
                scale1 = fs * 0.80

                def measure_row1(scale: float) -> tuple[int, int, int, int, int, int]:
                    widths = [
                        cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, bt)[0][0]
                        for text in (prefix, verdict, metrics, r_model, r_trust, r_stats)
                    ]
                    return tuple(widths)

                tw_p, tw_v, tw_m, tw_rm, tw_rt, tw_rs = measure_row1(scale1)
                total_w1 = (tw_p + tw_v + tw_m) + (tw_rm + tw_rt + tw_rs) + margin * 3
                while total_w1 > width and scale1 > 0.20:
                    scale1 -= 0.04
                    tw_p, tw_v, tw_m, tw_rm, tw_rt, tw_rs = measure_row1(scale1)
                    total_w1 = (tw_p + tw_v + tw_m) + (tw_rm + tw_rt + tw_rs) + margin * 3

                cur_x = margin
                cv2.putText(overlay, prefix, (cur_x, y1), cv2.FONT_HERSHEY_SIMPLEX, scale1, (200, 200, 200), bt, cv2.LINE_AA)
                cur_x += tw_p
                cv2.putText(overlay, verdict, (cur_x, y1), cv2.FONT_HERSHEY_SIMPLEX, scale1, health_accent, bt, cv2.LINE_AA)
                cur_x += tw_v
                cv2.putText(overlay, metrics, (cur_x, y1), cv2.FONT_HERSHEY_SIMPLEX, scale1, (240, 240, 240), bt, cv2.LINE_AA)

                rx = width - margin - (tw_rm + tw_rt + tw_rs)
                cv2.putText(overlay, r_model, (rx, y1), cv2.FONT_HERSHEY_SIMPLEX, scale1, (200, 200, 200), bt, cv2.LINE_AA)
                rx += tw_rm
                cv2.putText(overlay, r_trust, (rx, y1), cv2.FONT_HERSHEY_SIMPLEX, scale1, trust_accent, bt, cv2.LINE_AA)
                rx += tw_rt
                cv2.putText(overlay, r_stats, (rx, y1), cv2.FONT_HERSHEY_SIMPLEX, scale1, (200, 200, 200), bt, cv2.LINE_AA)
                bar_y = y1 + max(8, int(row_h * 0.82))

            # Row 2: Stacked Benthic Composition Progress Bar
            curr_x = margin
            for i, name in enumerate(self.class_names):
                count = counts.get(name, 0)
                frac = count / total_valid if total_valid > 0 else 0.0
                w_seg = int(round(frac * bar_w))
                if w_seg > 0:
                    x_end = min(curr_x + w_seg, margin + bar_w)
                    col = tuple(int(c) for c in self.colors[i])
                    cv2.rectangle(overlay, (curr_x, bar_y), (x_end, bar_y + bar_h), col, -1)
                    cv2.rectangle(overlay, (curr_x, bar_y), (x_end, bar_y + bar_h), (25, 25, 25), 1)
                    curr_x = x_end

            # Row 3: Color Guide Legend & Substrate Percentages
            legend_labels = {
                "coral_alive": f"Alive {proportions.get('coral_alive', 0.0) * 100:.1f}%",
                "coral_bleached": f"Bleached {proportions.get('coral_bleached', 0.0) * 100:.1f}%",
                "coral_dead": f"Dead {proportions.get('coral_dead', 0.0) * 100:.1f}%",
                "algae": f"Algae {proportions.get('algae', 0.0) * 100:.1f}%",
                "hard_substrate": f"Substrate {proportions.get('hard_substrate', 0.0) * 100:.1f}%",
                "sand": f"Sand {proportions.get('sand', 0.0) * 100:.1f}%",
                "non_reef": f"Non-Reef {proportions.get('non_reef', 0.0) * 100:.1f}%",
            }

            y3 = bar_y + bar_h + int(row_h * (0.8 if small_banner else 0.85))
            legend_scale = min(0.28, fs * 0.65) if small_banner else max(0.35, fs * 0.65)
            (_, text_h), _ = cv2.getTextSize("Alive", cv2.FONT_HERSHEY_SIMPLEX, legend_scale, 1)

            # Larger swatches vertically centered with the text letters
            swatch_sz = max(6, int(text_h * 1.15)) if small_banner else max(11, int(text_h * 1.15))
            y_mid = y3 - text_h // 2
            box_top = y_mid - swatch_sz // 2
            box_bot = box_top + swatch_sz

            step = bar_w // len(self.class_names)
            for idx, name in enumerate(self.class_names):
                lx = margin + idx * step
                col = tuple(int(c) for c in self.colors[idx])
                small_legend_labels = {
                    "coral_alive": "ALV", "coral_bleached": "BLE", "coral_dead": "DED",
                    "algae": "ALG", "hard_substrate": "SUB", "sand": "SND", "non_reef": "NR",
                }
                lbl = small_legend_labels.get(name, name) if small_banner else legend_labels.get(name, name)
                cv2.rectangle(overlay, (lx, box_top), (lx + swatch_sz, box_bot), col, -1)
                cv2.rectangle(overlay, (lx, box_top), (lx + swatch_sz, box_bot), (190, 190, 190), 1)
                cv2.putText(overlay, lbl, (lx + swatch_sz + 6, y3), cv2.FONT_HERSHEY_SIMPLEX, legend_scale, (220, 220, 220), 1, cv2.LINE_AA)

            if save_overlay:
                cv2.imwrite(str(save_overlay), overlay)

        report = {
            "model": self.model_path,
            "model_type": self.model_type,
            "model_task": self.model_task,
            "confidence_available": mean_conf is not None,
            "image": source_desc,
            "width": width,
            "height": height,
            "quality": quality,
            "coverage": round(coverage, 4),
            "mean_confidence": round(mean_conf, 4) if mean_conf is not None else None,
            "prediction_uncertainty": round(uncertainty, 4) if uncertainty is not None else None,
            "health_indicators": health,
            "benthic_composition": proportions,
            "detections_count": len(detections),
            "detections": detections,
            "reliability": decision,
            "overlay": str(save_overlay) if save_overlay else None,
        }
        if return_overlay:
            report["overlay_image"] = overlay if (save_overlay or return_overlay) else None
        return report

    def _fold_project_map(self, logits, height: int, width: int) -> tuple[np.ndarray, np.ndarray, float]:
        """Fold fine label logits into project classes."""
        import math

        probabilities = logits.float().softmax(dim=1)
        grouped = torch.zeros(
            (1, len(CLASSES) + 1, *probabilities.shape[-2:]),
            dtype=probabilities.dtype,
            device=probabilities.device,
        )
        group_ids = torch.as_tensor(self._dense_project_ids, dtype=torch.long, device=probabilities.device)
        grouped.scatter_add_(
            1,
            group_ids.view(1, -1, 1, 1).expand(1, -1, *probabilities.shape[-2:]),
            probabilities,
        )

        normalised = grouped / grouped.sum(dim=1, keepdim=True).clamp_min(1e-6)
        confidence, prediction = grouped.max(dim=1)
        log_p = normalised.clamp_min(1e-8).log()
        entropy = -(normalised * log_p).sum(dim=1) / math.log(grouped.shape[1])
        del probabilities, grouped, normalised, log_p

        recognised = prediction != len(CLASSES)
        uncertainty = float(entropy[recognised].mean()) if bool(recognised.any()) else 1.0

        project_map = prediction[0].to("cpu").to(torch.uint8).numpy()
        project_map[project_map == len(CLASSES)] = 255
        confidence_map = confidence[0].float().to("cpu").numpy()
        return (
            cv2.resize(project_map, (width, height), interpolation=cv2.INTER_NEAREST),
            cv2.resize(confidence_map, (width, height), interpolation=cv2.INTER_LINEAR),
            float(np.clip(uncertainty, 0.0, 1.0)),
        )

    def _predict_segformer(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Return project-class IDs, pixel confidence and entropy from SegFormer."""
        import torch.nn.functional as F

        height, width = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (1376, 768), interpolation=cv2.INTER_LINEAR)
        inputs = self.processor(image_rgb, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            logits = F.interpolate(logits.float(), size=(768, 1376), mode="bilinear", align_corners=False)
        return self._fold_project_map(logits, height, width)

    def _predict_dinov3(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Return project-class IDs, pixel confidence and entropy from DINOv3 + DPT."""
        from PIL import Image as PILImage

        height, width = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = PILImage.fromarray(image_rgb).resize((1376, 768), PILImage.Resampling.BILINEAR)
        pixel_values = self.processor(images=image, return_tensors="pt", do_resize=False)["pixel_values"]
        with torch.inference_mode():
            logits = self.model(pixel_values.to(self.device))
        return self._fold_project_map(logits, height, width)

    def infer_video(self, video_path: str | Path, output_dir: str | Path,
                    output_video_name: str = "annotated_survey.mp4",
                    every_n: int = 1, max_frames: int | None = None,
                    save_frames: bool = False,
                    progress_callback: VideoProgressCallback | None = None) -> dict:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        out_fps = float(fps if every_n == 1 else max(1.0, round(fps / every_n, 2)))

        total_to_process = (
            (total_video_frames + every_n - 1) // every_n
            if total_video_frames > 0 else 100
        )
        if max_frames:
            total_to_process = min(total_to_process, max_frames)

        callback_accepts_live_data = False
        if progress_callback is not None:
            try:
                inspect.signature(progress_callback).bind(0, total_to_process, None, {})
                callback_accepts_live_data = True
            except (TypeError, ValueError):
                # Keep supporting callbacks that only accept (current, total).
                pass

        temp_raw = output_dir / f"temp_{output_video_name}"
        final_video = output_dir / output_video_name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_raw), fourcc, out_fps, (width, height))

        frame_idx, reports = 0, []

        while cap.isOpened():
            if max_frames and len(reports) >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % every_n == 0:
                frame_overlay_path = output_dir / f"frame_{frame_idx:06d}_overlay.jpg" if save_frames else None
                report = self.infer(frame, save_overlay=frame_overlay_path, return_overlay=True)
                report["frame_index"] = frame_idx
                overlay_img = report.get("overlay_image")
                if overlay_img is not None:
                    writer.write(overlay_img)
                    del report["overlay_image"]
                reports.append(report)
                if progress_callback:
                    if callback_accepts_live_data:
                        progress_callback(len(reports), total_to_process, overlay_img, report)
                    else:
                        progress_callback(len(reports), total_to_process)
            frame_idx += 1
        cap.release()
        writer.release()

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin and temp_raw.exists() and temp_raw.stat().st_size > 0:
            cmd = [
                ffmpeg_bin, "-y", "-i", str(temp_raw),
                "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                "-pix_fmt", "yuv420p", str(final_video)
            ]
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                temp_raw.unlink(missing_ok=True)
            except Exception:
                if temp_raw.exists():
                    temp_raw.replace(final_video)
        else:
            if temp_raw.exists():
                temp_raw.replace(final_video)

        summary = aggregate_reports(reports)
        result = {
            "video": str(video_path),
            "annotated_video": str(final_video) if final_video.exists() else None,
            "sampled_frames": len(reports),
            "summary": summary,
            "frames": reports,
        }
        (output_dir / "report.json").write_text(json.dumps(result, indent=2))
        return result


def aggregate_reports(reports: list[dict]) -> dict:
    """Aggregate frame-level reports into a survey-level health assessment."""
    if not reports:
        return {"status": "needs_review", "reason": "no_frames", "health_indicators": {}}

    accepted = [r for r in reports if r["reliability"]["status"] == "accepted"]
    pool = accepted if accepted else reports

    composition_names = list(pool[0]["benthic_composition"])
    mean_composition = {
        name: float(np.mean([r["benthic_composition"].get(name, 0.0) for r in pool]))
        for name in composition_names
    }

    mean_lcc = float(np.mean([r["health_indicators"]["live_coral_cover_pct"] for r in pool]))
    mean_bleached = float(np.mean([r["health_indicators"]["bleached_coral_cover_pct"] for r in pool]))
    mean_algae = float(np.mean([
        r["health_indicators"].get("algae_covered_cover_pct", r["health_indicators"]["macroalgae_cover_pct"])
        for r in pool
    ]))
    mean_bleaching_ratio = float(np.mean([r["health_indicators"]["bleaching_ratio_pct"] for r in pool]))
    mean_mortality_ratio = float(np.mean([r["health_indicators"]["mortality_ratio_pct"] for r in pool]))

    overall_cat, overall_cover_tier, overall_condition, _ = _reef_health_verdict(
        mean_lcc, mean_bleaching_ratio, mean_mortality_ratio, mean_algae, True
    )

    return {
        "status": "accepted" if accepted else "needs_review",
        "accepted_frames": len(accepted),
        "total_frames": len(reports),
        "accepted_frame_ratio": round(len(accepted) / len(reports), 4),
        "overall_health_category": overall_cat,
        "overall_live_cover_tier": overall_cover_tier,
        "overall_coral_condition": overall_condition,
        "mean_benthic_composition": {k: round(v, 4) for k, v in mean_composition.items()},
        "aggregated_health_metrics": {
            "mean_live_coral_cover_pct": round(mean_lcc, 2),
            "mean_bleached_coral_cover_pct": round(mean_bleached, 2),
            "mean_macroalgae_cover_pct": round(mean_algae, 2),
            "mean_algae_covered_cover_pct": round(mean_algae, 2),
            "mean_bleaching_ratio_pct": round(mean_bleaching_ratio, 2),
            "mean_mortality_ratio_pct": round(mean_mortality_ratio, 2),
        },
    }
