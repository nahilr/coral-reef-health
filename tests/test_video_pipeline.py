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


def test_video_pipeline_live_callback() -> None:
    """Verify that infer_video invokes 4-arg live callbacks with frames and reports."""
    model_path = Path("models/checkpoints/yolo26n_sem/weights/best.pt")
    if not model_path.exists():
        model_path = Path("models/checkpoints/yolo26n_seg/weights/best.pt")

    pipeline = CoralPipeline(model_path, imgsz=320)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        input_vid = tmp_dir / "input.mp4"
        out_dir = tmp_dir / "out"

        create_dummy_video(input_vid, num_frames=6, fps=10.0)

        received_calls = []

        def _live_cb(current: int, total: int, overlay_bgr: np.ndarray | None = None, report: dict | None = None) -> None:
            received_calls.append({
                "current": current,
                "total": total,
                "has_overlay": overlay_bgr is not None,
                "overlay_shape": overlay_bgr.shape if overlay_bgr is not None else None,
                "has_health": "health_indicators" in report if report else False,
                "has_reliability": "reliability" in report if report else False,
            })

        pipeline.infer_video(
            video_path=input_vid,
            output_dir=out_dir,
            output_video_name="annotated_live.mp4",
            every_n=1,
            max_frames=4,
            progress_callback=_live_cb,
        )

        assert len(received_calls) == 4
        assert [c["current"] for c in received_calls] == [1, 2, 3, 4]
        for c in received_calls:
            assert c["has_overlay"] is True
            assert c["overlay_shape"] == (240, 320, 3)
            assert c["has_health"] is True
            assert c["has_reliability"] is True


def test_video_pipeline_fallback_callback() -> None:
    """Verify that infer_video maintains backward compatibility for legacy 2-arg callbacks."""
    model_path = Path("models/checkpoints/yolo26n_sem/weights/best.pt")
    if not model_path.exists():
        model_path = Path("models/checkpoints/yolo26n_seg/weights/best.pt")

    pipeline = CoralPipeline(model_path, imgsz=320)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        input_vid = tmp_dir / "input.mp4"
        out_dir = tmp_dir / "out"

        create_dummy_video(input_vid, num_frames=4, fps=10.0)

        two_arg_calls = []

        def _legacy_cb(current: int, total: int) -> None:
            two_arg_calls.append((current, total))

        pipeline.infer_video(
            video_path=input_vid,
            output_dir=out_dir,
            output_video_name="annotated_legacy.mp4",
            every_n=1,
            max_frames=3,
            progress_callback=_legacy_cb,
        )

        assert len(two_arg_calls) == 3
        assert two_arg_calls == [(1, 3), (2, 3), (3, 3)]
