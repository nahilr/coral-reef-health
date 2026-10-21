#!/usr/bin/env python3
"""Single entry point for the coral-reef health assessment pipeline.

Subcommands:
  prepare          prepare the Coralscapes project-taxonomy YOLO dataset
  evaluate         evaluate a segmentation checkpoint on the held-out test split
  infer            run image or video inference
  app              launch the Streamlit UI
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    p = argparse.ArgumentParser(prog="main.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare", help="prepare Coralscapes as project-taxonomy YOLO data")
    prep.add_argument("--root", type=Path, default=Path("data/raw/coralscapes"))
    prep.add_argument("--out", type=Path, default=Path("data/processed/yolo"))
    prep.add_argument("--max-images-per-split", type=int, default=None)

    ev = sub.add_parser("evaluate", help="evaluate segmentation checkpoint")
    ev.add_argument("--model", default="models/checkpoints/yolo26n_sem/weights/best.pt")
    ev.add_argument("--manifest", default="data/processed/yolo/manifest.csv")
    ev.add_argument("--raw-classes", default="configs/dinov3_id2label.json")
    ev.add_argument("--max-images", type=int, default=None)
    ev.add_argument("--conf", type=float, default=0.20)
    ev.add_argument("--imgsz", type=int, default=512)
    ev.add_argument("--out", default="reports/eval_metrics.json")

    inf = sub.add_parser("infer", help="image or video inference")
    inf.add_argument("--model", default="models/checkpoints/yolo26n_sem/weights/best.pt")
    inf.add_argument("--model-type", choices=["yolo", "segformer", "dinov3"], default="yolo")
    inf.add_argument("--image")
    inf.add_argument("--video")
    inf.add_argument("--out", default="outputs/inference")
    inf.add_argument("--imgsz", type=int, default=512)
    inf.add_argument("--conf", type=float, default=0.20)
    inf.add_argument("--every-n", type=int, default=1, help="Process every N-th frame (default: 1)")
    inf.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process (default: all)")
    inf.add_argument("--save-frames", action="store_true", help="Also save individual overlay JPEG frames")
    inf.add_argument("--output-video-name", default="annotated_survey.mp4", help="Name of output video file")

    app = sub.add_parser("app", help="start the Streamlit UI")
    app.add_argument("--port", type=int, default=8501)

    args = p.parse_args()

    if args.cmd == "prepare":
        from src.data.prepare import prepare
        print(json.dumps(prepare(args.root, args.out, args.max_images_per_split), indent=2))
    elif args.cmd == "evaluate":
        from src.evaluate.segmentation import evaluate
        print(json.dumps(evaluate(args.model, args.manifest, args.raw_classes,
                                  max_images=args.max_images, conf=args.conf, imgsz=args.imgsz, out=args.out), indent=2))
    elif args.cmd == "infer":
        if not args.image and not args.video:
            p.error("Must specify either --image or --video")
        from src.inference.pipeline import CoralPipeline
        pipeline = CoralPipeline(args.model, imgsz=args.imgsz, conf=args.conf, model_type=args.model_type)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        if args.image:
            report = pipeline.infer(args.image, out / "overlay.jpg")
            (out / "report.json").write_text(json.dumps(report, indent=2))
        else:
            report = pipeline.infer_video(args.video, args.out,
                                          output_video_name=args.output_video_name,
                                          every_n=args.every_n,
                                          max_frames=args.max_frames,
                                          save_frames=args.save_frames)
            if report.get("annotated_video"):
                print(f"Annotated video saved to: {report['annotated_video']}")
        print(json.dumps(report, indent=2))
    elif args.cmd == "app":
        cmd = [sys.executable, "-m", "streamlit", "run", "src/app/streamlit_app.py", "--server.port", str(args.port)]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
