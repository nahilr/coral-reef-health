from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

from src.data.label_map import BROAD_CLASSES
from src.data.quality import assess_image
from src.trust.reliability import normalized_entropy, review_decision

COLORS = np.array([[70, 170, 70], [255, 230, 80], [150, 80, 80], [50, 160, 50], [150, 120, 70], [190, 170, 120], [120, 120, 120]], dtype=np.uint8)


class CoralPipeline:
    def __init__(self, model_path: str | Path, imgsz: int = 384, conf: float = 0.20):
        self.model_path, self.imgsz, self.conf = str(model_path), imgsz, conf
        self.model = YOLO(self.model_path)

    def infer(self, image_path: str | Path, save_overlay: str | Path | None = None) -> dict:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None: raise ValueError(f"Cannot read image: {image_path}")
        quality = assess_image(image)
        result = self.model.predict(source=image, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        height, width = image.shape[:2]
        class_map = np.full((height, width), -1, dtype=np.int16)
        confidence_map = np.zeros((height, width), dtype=np.float32)
        detections = []
        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            scores = result.boxes.conf.detach().cpu().numpy()
            for mask, cls, score in zip(masks, classes, scores):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 0.5
                replace = mask & (score >= confidence_map)
                class_map[replace], confidence_map[replace] = cls, score
                detections.append({"class": BROAD_CLASSES[cls] if cls < len(BROAD_CLASSES) else str(cls), "confidence": float(score), "pixels": int(mask.sum())})
        valid = class_map >= 0
        counts = {name: int((class_map == i).sum()) for i, name in enumerate(BROAD_CLASSES)}
        total = int(valid.sum())
        proportions = {name: (counts[name] / total if total else 0.0) for name in BROAD_CLASSES}
        mean_conf = float(confidence_map[valid].mean()) if valid.any() else 0.0
        class_probs = np.array([proportions[name] for name in BROAD_CLASSES], dtype=float)
        entropy = normalized_entropy(class_probs / class_probs.sum()) if class_probs.sum() else 1.0
        coverage = total / float(width * height)
        decision = review_decision(quality, mean_conf, coverage, entropy)
        if save_overlay:
            overlay = image.copy()
            for cls, color in enumerate(COLORS):
                sel = class_map == cls
                overlay[sel] = (0.55 * overlay[sel] + 0.45 * color).astype(np.uint8)
            # Draw bounding boxes with class + confidence labels.
            fs = max(0.6, min(2.2, width / 700.0))  # font scale relative to image size
            bt = max(2, int(round(width / 400)))     # box thickness
            if result.masks is not None and result.boxes is not None:
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                classes = result.boxes.cls.detach().cpu().numpy().astype(int)
                scores = result.boxes.conf.detach().cpu().numpy()
                for (x1, y1, x2, y2), cls, score in zip(boxes, classes, scores):
                    color = tuple(int(c) for c in COLORS[cls]) if cls < len(COLORS) else (200, 200, 200)
                    cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, bt)
                    name = BROAD_CLASSES[cls] if cls < len(BROAD_CLASSES) else str(cls)
                    label = f"{name} {score:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, bt)
                    cv2.rectangle(overlay, (int(x1), int(y1) - th - 8), (int(x1) + tw + 8, int(y1)), color, -1)
                    cv2.putText(overlay, label, (int(x1) + 4, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), bt, cv2.LINE_AA)
            # Panel with composition percentages and reliability status.
            lines = [f"coverage {coverage*100:.1f}%  conf {mean_conf:.2f}  {decision['status']}"]
            lines += [f"{name}: {proportions[name]*100:.1f}%" for name in BROAD_CLASSES if proportions[name] > 0]
            panel_w = int(width * 0.38)
            row_h = int(26 * fs) + 6
            panel = np.zeros((len(lines) * row_h + 12, panel_w, 3), dtype=np.uint8)
            for i, line in enumerate(lines):
                cv2.putText(panel, line, (10, row_h * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), bt, cv2.LINE_AA)
            panel_h = panel.shape[0]
            overlay[:panel_h, :panel_w] = (0.65 * overlay[:panel_h, :panel_w] + 0.35 * panel).astype(np.uint8)
            cv2.imwrite(str(save_overlay), overlay)
        return {"model": self.model_path, "image": str(image_path), "width": width, "height": height,
                "quality": quality, "coverage": coverage, "mean_confidence": mean_conf,
                "entropy": entropy, "composition": proportions, "detections": detections,
                "reliability": decision, "overlay": str(save_overlay) if save_overlay else None}

    def infer_video(self, video_path: str | Path, output_dir: str | Path, every_n: int = 15, max_frames: int = 30) -> dict:
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        cap, frame_idx, reports = cv2.VideoCapture(str(video_path)), 0, []
        while len(reports) < max_frames:
            ok, frame = cap.read()
            if not ok: break
            if frame_idx % every_n == 0:
                frame_path, overlay_path = output_dir / f"frame_{frame_idx:06d}.jpg", output_dir / f"frame_{frame_idx:06d}_overlay.jpg"
                cv2.imwrite(str(frame_path), frame)
                report = self.infer(frame_path, overlay_path); report["frame_index"] = frame_idx; reports.append(report)
            frame_idx += 1
        cap.release()
        result = {"video": str(video_path), "sampled_frames": len(reports), "frames": reports, "summary": aggregate_reports(reports)}
        (output_dir / "report.json").write_text(json.dumps(result, indent=2))
        return result


def aggregate_reports(reports: list[dict]) -> dict:
    if not reports: return {"status": "needs_review", "reason": "no_frames"}
    accepted = [r for r in reports if r["reliability"]["status"] == "accepted"]
    pool = accepted or reports
    composition = {name: float(np.mean([r["composition"][name] for r in pool])) for name in BROAD_CLASSES}
    return {"accepted_frames": len(accepted), "total_frames": len(reports), "accepted_frame_coverage": len(accepted) / len(reports), "composition": composition, "status": "accepted" if accepted else "needs_review"}
