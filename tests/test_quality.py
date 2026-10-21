"""Unit tests for image quality assessment gate."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.quality import assess_image


def test_quality_path_and_array_consistency() -> None:
    """Assessments from file path and OpenCV BGR numpy array must be identical."""
    # Create synthetic image with distinct B, G, R channels
    # Blue=180, Green=100, Red=40
    img_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    img_bgr[:, :, 0] = 180
    img_bgr[:, :, 1] = 100
    img_bgr[:, :, 2] = 40

    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        cv2.imwrite(tmp.name, img_bgr)
        res_from_path = assess_image(tmp.name)
        res_from_array = assess_image(img_bgr)

        assert abs(res_from_path["brightness_mean"] - res_from_array["brightness_mean"]) < 1e-4
        assert abs(res_from_path["contrast_std"] - res_from_array["contrast_std"]) < 1e-4
        assert abs(res_from_path["blur_score"] - res_from_array["blur_score"]) < 1e-4
        assert res_from_path["flags"] == res_from_array["flags"]
        assert res_from_path["status"] == res_from_array["status"]


def test_empty_or_invalid_image() -> None:
    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    res = assess_image(empty_img)
    assert res["status"] == "needs_review"
    assert "unreadable" in res["flags"]


def test_exposure_flags() -> None:
    # Completely dark image
    dark_img = np.full((100, 100, 3), 10, dtype=np.uint8)
    res_dark = assess_image(dark_img)
    assert "exposure" in res_dark["flags"]
    assert res_dark["status"] == "needs_review"

    # Overexposed bright image
    bright_img = np.full((100, 100, 3), 245, dtype=np.uint8)
    res_bright = assess_image(bright_img)
    assert "exposure" in res_bright["flags"]
    assert res_bright["status"] == "needs_review"


def test_contrast_and_blur_flags() -> None:
    # Completely flat image (contrast std = 0, blur = 0)
    flat_img = np.full((100, 100, 3), 128, dtype=np.uint8)
    res_flat = assess_image(flat_img)
    assert "low_contrast" in res_flat["flags"]
    assert "blur" in res_flat["flags"]
    assert res_flat["status"] == "needs_review"


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} quality tests passed")
