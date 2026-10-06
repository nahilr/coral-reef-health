from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.data.label_map import BROAD_CLASSES
from src.data.quality import assess_image
from src.trust.reliability import prediction_uncertainty, review_decision

COLORS = np.array([
    [70, 170, 70],    # coral_alive (green)
    [255, 230, 80],   # coral_bleached (yellow)
    [150, 80, 80],    # coral_dead (dark red/brown)
    [50, 160, 50],    # algae (forest green)
    [150, 120, 70],   # rubble (brown)
    [190, 170, 120],  # sand_rock (sand)
    [120, 120, 120],  # other (gray)
], dtype=np.uint8)


def compute_health_indicators(counts: dict[str, int], total_valid: int) -> dict:
    """Compute standard quantitative reef health metrics from segmented benthic pixels."""
    if total_valid <= 0:
        return {
            "live_coral_cover_pct": 0.0,
            "bleached_coral_cover_pct": 0.0,
            "dead_coral_cover_pct": 0.0,
            "macroalgae_cover_pct": 0.0,
            "rubble_cover_pct": 0.0,
            "sand_rock_cover_pct": 0.0,
            "total_coral_cover_pct": 0.0,
            "bleaching_ratio_pct": 0.0,
            "mortality_ratio_pct": 0.0,
            "coral_to_algae_ratio": 0.0,
            "reef_health_category": "Indeterminate (No benthic coverage detected)",
        }

    live = counts.get("coral_alive", 0)
    bleached = counts.get("coral_bleached", 0)
    dead = counts.get("coral_dead", 0)
    algae = counts.get("algae", 0)
    rubble = counts.get("rubble", 0)
    sand = counts.get("sand_rock", 0)

    total_coral = live + bleached + dead
    lcc = (live / total_valid) * 100.0
    bleached_cover = (bleached / total_valid) * 100.0
    dead_cover = (dead / total_valid) * 100.0
    algae_cover = (algae / total_valid) * 100.0
    rubble_cover = (rubble / total_valid) * 100.0
    sand_cover = (sand / total_valid) * 100.0
    total_coral_cover = (total_coral / total_valid) * 100.0

    bleaching_ratio = (bleached / total_coral * 100.0) if total_coral > 0 else 0.0
    mortality_ratio = (dead / total_coral * 100.0) if total_coral > 0 else 0.0
    algae_ratio = float(algae / (live + 1e-4))

    # Standard ecological condition tiers (Reef Check / Gomez et al. benchmark)
    if lcc >= 50.0:
        cat = "Excellent"
    elif lcc >= 25.0:
        cat = "Good"
    elif lcc >= 10.0:
        cat = "Fair"
    else:
        cat = "Poor / Degraded"

    if bleaching_ratio >= 25.0:
        cat += " [Severe Bleaching Alert]"
    elif bleaching_ratio >= 10.0:
        cat += " [Moderate Bleaching Warning]"
    elif algae_cover > 30.0 and algae_cover > lcc:
        cat += " [Macroalgal Competition Warning]"

    return {
        "live_coral_cover_pct": round(lcc, 2),
        "bleached_coral_cover_pct": round(bleached_cover, 2),
        "dead_coral_cover_pct": round(dead_cover, 2),
        "macroalgae_cover_pct": round(algae_cover, 2),
        "rubble_cover_pct": round(rubble_cover, 2),
        "sand_rock_cover_pct": round(sand_cover, 2),
        "total_coral_cover_pct": round(total_coral_cover, 2),
        "bleaching_ratio_pct": round(bleaching_ratio, 2),
        "mortality_ratio_pct": round(mortality_ratio, 2),
        "coral_to_algae_ratio": round(algae_ratio, 2),
        "reef_health_category": cat,
    }


class CoralPipeline:
    def __init__(self, model_path: str | Path, condition_model_path: str | Path | None = None,
                 imgsz: int = 512, conf: float = 0.20, thresholds: dict | None = None):
        self.model_path = str(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.thresholds = thresholds
        self.model = YOLO(self.model_path)

        # Optional condition classifier (ResNet18) for crop-level verification
        self.condition_model = None
        self.condition_classes = []
        cond_path = Path(condition_model_path) if condition_model_path else Path("models/checkpoints/condition_resnet18.pt")
        if cond_path.exists():
            try:
                from torchvision import models
                from torchvision import transforms
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                ckpt = torch.load(cond_path, map_location=device, weights_only=False)
                classes = ckpt.get("classes", ["coral_alive", "coral_bleached", "coral_dead"])
                net = models.resnet18(weights=None)
                net.fc = torch.nn.Linear(net.fc.in_features, len(classes))
                net.load_state_dict(ckpt["state_dict"])
                net.to(device).eval()
                self.condition_model = net
                self.condition_classes = classes
                self.condition_device = device
                self.condition_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
            except Exception as e:
                self.condition_model = None

    def infer(self, image_path: str | Path, save_overlay: str | Path | None = None) -> dict:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        quality = assess_image(image)
        result = self.model.predict(source=image, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]

        height, width = image.shape[:2]
        class_map = np.full((height, width), -1, dtype=np.int16)
        confidence_map = np.zeros((height, width), dtype=np.float32)
        detections = []
        crop_evaluations = []

        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            scores = result.boxes.conf.detach().cpu().numpy()
            boxes = result.boxes.xyxy.detach().cpu().numpy()

            for idx, (mask, cls, score, (x1, y1, x2, y2)) in enumerate(zip(masks, classes, scores, boxes)):
                mask_full = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 0.5
                replace = mask_full & (score >= confidence_map)
                class_map[replace] = cls
                confidence_map[replace] = score

                cls_name = BROAD_CLASSES[cls] if cls < len(BROAD_CLASSES) else str(cls)
                det_info = {
                    "id": idx,
                    "class": cls_name,
                    "confidence": float(score),
                    "pixels": int(mask_full.sum()),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }

                # If this is a coral detection and we have the condition classifier, classify crop
                if self.condition_model is not None and "coral" in cls_name:
                    bx1, by1 = max(0, int(x1)), max(0, int(y1))
                    bx2, by2 = min(width, int(x2)), min(height, int(y2))
                    if (bx2 - bx1) >= 16 and (by2 - by1) >= 16:
                        crop_rgb = cv2.cvtColor(image[by1:by2, bx1:bx2], cv2.COLOR_BGR2RGB)
                        inp = self.condition_transform(crop_rgb).unsqueeze(0).to(self.condition_device)
                        with torch.no_grad():
                            logits = self.condition_model(inp)
                            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                            pred_idx = int(probs.argmax())
                            pred_cond = self.condition_classes[pred_idx]
                            det_info["condition_crop_pred"] = pred_cond
                            det_info["condition_crop_confidence"] = float(probs[pred_idx])
                            crop_evaluations.append({
                                "detection_id": idx,
                                "predicted_condition": pred_cond,
                                "probabilities": {c: float(p) for c, p in zip(self.condition_classes, probs)},
                            })

                detections.append(det_info)

        valid = class_map >= 0
        total_valid = int(valid.sum())
        counts = {name: int((class_map == i).sum()) for i, name in enumerate(BROAD_CLASSES)}
        proportions = {name: round(counts[name] / total_valid, 4) if total_valid else 0.0 for name in BROAD_CLASSES}

        mean_conf = float(confidence_map[valid].mean()) if valid.any() else 0.0
        coverage = total_valid / float(width * height)

        # Compute predictive uncertainty from detection confidence scores
        det_scores = [d["confidence"] for d in detections]
        uncertainty = prediction_uncertainty(det_scores)

        # Health indicators
        health = compute_health_indicators(counts, total_valid)

        # Trust / Reliability gate
        decision = review_decision(quality, mean_conf, coverage, uncertainty, thresholds=self.thresholds)

        if save_overlay:
            overlay = image.copy()
            # Semi-transparent mask overlays
            for cls_idx, color in enumerate(COLORS):
                sel = class_map == cls_idx
                if sel.any():
                    overlay[sel] = (0.55 * overlay[sel] + 0.45 * color).astype(np.uint8)

            # Draw bounding boxes with class + confidence labels
            fs = max(0.45, min(1.2, width / 1200.0))
            bt = max(1, int(round(width / 600)))
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cls_name = det["class"]
                score = det["confidence"]
                cls_idx = BROAD_CLASSES.index(cls_name) if cls_name in BROAD_CLASSES else 6
                color = tuple(int(c) for c in COLORS[cls_idx])

                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, bt)
                label = f"{cls_name} {score:.2f}"
                if "condition_crop_pred" in det:
                    label += f" [{det['condition_crop_pred'][:5]}]"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, bt)
                cv2.rectangle(overlay, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(0, y1)), color, -1)
                cv2.putText(overlay, label, (x1 + 3, max(th, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), bt, cv2.LINE_AA)

            # Informational HUD panel
            status_text = decision["status"].upper()
            status_color = (60, 180, 60) if decision["status"] == "accepted" else (40, 100, 220)

            hud_lines = [
                f"REEF HEALTH: {health['reef_health_category']}",
                f"LCC: {health['live_coral_cover_pct']}% | Bleached: {health['bleached_coral_cover_pct']}% | Algae: {health['macroalgae_cover_pct']}%",
                f"TRUST STATUS: {status_text} | Coverage: {coverage*100:.1f}% | Conf: {mean_conf:.2f}",
            ]
            if decision["reasons"]:
                hud_lines.append(f"FLAGS: {', '.join(decision['reasons'])}")

            panel_w = max(420, int(width * 0.42))
            row_h = int(22 * fs) + 8
            panel_h = len(hud_lines) * row_h + 16
            panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

            for i, line in enumerate(hud_lines):
                c = (255, 255, 255) if i != 2 else status_color
                cv2.putText(panel, line, (12, row_h * (i + 1) - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, c, bt, cv2.LINE_AA)

            overlay[:panel_h, :panel_w] = (0.50 * overlay[:panel_h, :panel_w] + 0.50 * panel).astype(np.uint8)
            cv2.imwrite(str(save_overlay), overlay)

        report = {
            "model": self.model_path,
            "image": str(image_path),
            "width": width,
            "height": height,
            "quality": quality,
            "coverage": round(coverage, 4),
            "mean_confidence": round(mean_conf, 4),
            "prediction_uncertainty": round(uncertainty, 4),
            "health_indicators": health,
            "benthic_composition": proportions,
            "detections_count": len(detections),
            "detections": detections,
            "condition_classifier_evaluated": len(crop_evaluations) > 0,
            "crop_evaluations": crop_evaluations,
            "reliability": decision,
            "overlay": str(save_overlay) if save_overlay else None,
        }
        return report

    def infer_video(self, video_path: str | Path, output_dir: str | Path,
                    every_n: int = 15, max_frames: int = 30) -> dict:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        frame_idx, reports = 0, []

        while len(reports) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % every_n == 0:
                frame_path = output_dir / f"frame_{frame_idx:06d}.jpg"
                overlay_path = output_dir / f"frame_{frame_idx:06d}_overlay.jpg"
                cv2.imwrite(str(frame_path), frame)
                report = self.infer(frame_path, overlay_path)
                report["frame_index"] = frame_idx
                reports.append(report)
            frame_idx += 1
        cap.release()

        summary = aggregate_reports(reports)
        result = {
            "video": str(video_path),
            "sampled_frames": len(reports),
            "summary": summary,
            "frames": reports,
        }
        (output_dir / "report.json").write_text(json.dumps(result, indent=2))
        return result


def aggregate_reports(reports: list[dict]) -> dict:
    """Aggregate frame-level reports into a video/survey-level health assessment."""
    if not reports:
        return {"status": "needs_review", "reason": "no_frames", "health_indicators": {}}

    accepted = [r for r in reports if r["reliability"]["status"] == "accepted"]
    pool = accepted if accepted else reports

    mean_composition = {
        name: float(np.mean([r["benthic_composition"][name] for r in pool]))
        for name in BROAD_CLASSES
    }

    # Aggregate health metrics
    mean_lcc = float(np.mean([r["health_indicators"]["live_coral_cover_pct"] for r in pool]))
    mean_bleached = float(np.mean([r["health_indicators"]["bleached_coral_cover_pct"] for r in pool]))
    mean_algae = float(np.mean([r["health_indicators"]["macroalgae_cover_pct"] for r in pool]))
    mean_bleaching_ratio = float(np.mean([r["health_indicators"]["bleaching_ratio_pct"] for r in pool]))

    if mean_lcc >= 50.0:
        overall_cat = "Excellent"
    elif mean_lcc >= 25.0:
        overall_cat = "Good"
    elif mean_lcc >= 10.0:
        overall_cat = "Fair"
    else:
        overall_cat = "Poor / Degraded"

    if mean_bleaching_ratio >= 20.0:
        overall_cat += " [Active Bleaching Detected Across Video]"

    return {
        "status": "accepted" if accepted else "needs_review",
        "accepted_frames": len(accepted),
        "total_frames": len(reports),
        "accepted_frame_ratio": round(len(accepted) / len(reports), 4),
        "overall_health_category": overall_cat,
        "mean_benthic_composition": {k: round(v, 4) for k, v in mean_composition.items()},
        "aggregated_health_metrics": {
            "mean_live_coral_cover_pct": round(mean_lcc, 2),
            "mean_bleached_coral_cover_pct": round(mean_bleached, 2),
            "mean_macroalgae_cover_pct": round(mean_algae, 2),
            "mean_bleaching_ratio_pct": round(mean_bleaching_ratio, 2),
        },
    }
