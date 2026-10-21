"""Regression tests for video-to-video pipeline inference."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.pipeline import CoralPipeline


def create_dummy_video(path: Path, num_frames: int = 6, width: int = 320, height: int = 240, fps: float = 10.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        # Create a simple synthetic RGB frame with varying green/blue tones
        frame = np.full((height, width, 3), (120 + i * 5, 80, 50), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_video_to_video_generation() -> None:
    model_path = Path("models/checkpoints/yolo26n_sem/weights/best.pt")
    if not model_path.exists():
        model_path = Path("models/checkpoints/yolo26n_seg/weights/best.pt")

    pipeline = CoralPipeline(model_path, imgsz=320)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        input_vid = tmp_dir / "input.mp4"
        out_dir = tmp_dir / "out"

        create_dummy_video(input_vid, num_frames=6, fps=10.0)

        result = pipeline.infer_video(
            video_path=input_vid,
            output_dir=out_dir,
            output_video_name="annotated_output.mp4",
            every_n=1,
            max_frames=4,
        )

        assert result["sampled_frames"] == 4
        assert "summary" in result
        assert "overall_health_category" in result["summary"]
        assert result.get("annotated_video") is not None

        out_vid_path = Path(result["annotated_video"])
        assert out_vid_path.exists()
        assert out_vid_path.stat().st_size > 0
