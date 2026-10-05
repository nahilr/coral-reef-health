from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def assess_image(image_or_path, min_blur: float = 12.0) -> dict:
    if isinstance(image_or_path, (str, Path)):
        image = cv2.imread(str(image_or_path), cv2.IMREAD_COLOR)
    else:
        image = np.asarray(image_or_path)
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if image is None or image.size == 0:
        return {"quality_score": 0.0, "status": "needs_review", "flags": ["unreadable"]}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean = float(gray.mean())
    contrast = float(gray.std())
    channel_means = image.reshape(-1, 3).mean(axis=0)
    color_cast = float((channel_means.max() + 1e-6) / (channel_means.min() + 1e-6)) > 2.2
    flags = []
    if blur < min_blur: flags.append("blur")
    if mean < 25 or mean > 235: flags.append("exposure")
    if contrast < 18: flags.append("low_contrast")
    if color_cast: flags.append("color_cast")
    score = 1.0 - (0.35 if "blur" in flags else 0) - (0.25 if "exposure" in flags else 0)
    score -= 0.20 if "low_contrast" in flags else 0
    score -= 0.20 if "color_cast" in flags else 0
    return {"blur_score": blur, "brightness_mean": mean, "contrast_std": contrast,
            "color_cast": color_cast, "quality_score": max(0.0, round(score, 4)),
            "status": "needs_review" if flags else "accepted", "flags": flags}
