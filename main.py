#!/usr/bin/env python3
"""Single entry point for the coral-reef trust pipeline.

Subcommands:
  prepare          prepare the legacy V1 YOLO seg dataset from a local snapshot
  prepare-v2       prepare the CoralscapesV2 project-taxonomy YOLO dataset
  evaluate         evaluate the legacy V1 checkpoint
  evaluate-v2      evaluate the V2 checkpoint on its held-out test split
  infer            run image inference for a single file or video
  train-condition  extract coral crops and train the condition classifier
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

    prep = sub.add_parser("prepare", help="prepare YOLO-format dataset")
    prep.add_argument("--root", type=Path, default=Path("data/raw/coralscapes_hf"))
    prep.add_argument("--out", type=Path, default=Path("data/processed/v1_yolo"))
    prep.add_argument("--max-images-per-split", type=int, default=None)
    prep.add_argument("--copy-images", action="store_true")

    prep_v2 = sub.add_parser("prepare-v2", help="prepare CoralscapesV2 as project-taxonomy YOLO data")
    prep_v2.add_argument("--root", type=Path, default=Path("data/raw/coralscapes_v2"))
    prep_v2.add_argument("--out", type=Path, default=Path("data/processed/v2_yolo"))
    prep_v2.add_argument("--max-images-per-split", type=int, default=None)

    ev = sub.add_parser("evaluate", help="evaluate segmentation checkpoint")
    ev.add_argument("--model", default="models/checkpoints/yolo26n_seg/weights/best.pt")
    ev.add_argument("--manifest", default="data/processed/v1_yolo/manifest.csv")
    ev.add_argument("--raw-classes", default="data/processed/v1_yolo/classes.json")
    ev.add_argument("--max-images", type=int, default=None)
    ev.add_argument("--conf", type=float, default=0.20)
    ev.add_argument("--out", default="reports/segmentation_metrics.json")

    ev2 = sub.add_parser("evaluate-v2", help="evaluate the CoralscapesV2 project-taxonomy checkpoint")
    ev2.add_argument("--model", default="models/checkpoints/yolo26n_sem_v2/weights/best.pt")
    ev2.add_argument("--manifest", default="data/processed/v2_yolo/manifest.csv")
    ev2.add_argument("--raw-classes", default="data/raw/coralscapes_v2/id2label.json")
    ev2.add_argument("--max-images", type=int, default=None)
    ev2.add_argument("--conf", type=float, default=0.20)
    ev2.add_argument("--out", default="reports/coralscapes_v2_metrics.json")

    inf = sub.add_parser("infer", help="image or video inference")
    inf.add_argument("--model", default="models/checkpoints/yolo26n_seg/weights/best.pt")
    inf.add_argument("--model-type", choices=["yolo", "segformer", "dinov3"], default="yolo")
    inf.add_argument("--condition-model", default=None)
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

    tc = sub.add_parser("train-condition", help="extract coral crops and train condition classifier")
    tc.add_argument("--manifest", default="data/processed/v1_yolo/manifest.csv")
    tc.add_argument("--raw-classes", default="data/processed/v1_yolo/classes.json")
    tc.add_argument("--crops", default="data/processed/v1_condition_crops")
    tc.add_argument("--checkpoint", default="models/checkpoints/condition_resnet18.pt")
    tc.add_argument("--per-class", type=int, default=250)
    tc.add_argument("--epochs", type=int, default=1)
    tc.add_argument("--batch", type=int, default=32)

    args = p.parse_args()

    if args.cmd == "prepare":
        from src.data.prepare import prepare
        print(json.dumps(prepare(args.root, args.out, args.max_images_per_split, args.copy_images), indent=2))
    elif args.cmd == "prepare-v2":
        from src.data.prepare import prepare_v2
        print(json.dumps(prepare_v2(args.root, args.out, args.max_images_per_split), indent=2))
    elif args.cmd == "evaluate":
        from src.evaluate.segmentation import evaluate
        print(json.dumps(evaluate(args.model, args.manifest, args.raw_classes,
                                  max_images=args.max_images, conf=args.conf, out=args.out), indent=2))
    elif args.cmd == "evaluate-v2":
        from src.evaluate.segmentation import evaluate_v2
        print(json.dumps(evaluate_v2(args.model, args.manifest, args.raw_classes,
                                     max_images=args.max_images, conf=args.conf, out=args.out), indent=2))
    elif args.cmd == "infer":
        if not args.image and not args.video:
            p.error("Must specify either --image or --video")
        from src.inference.pipeline import CoralPipeline
        pipeline = CoralPipeline(args.model, condition_model_path=args.condition_model,
                                 imgsz=args.imgsz, conf=args.conf, model_type=args.model_type)
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
    elif args.cmd == "train-condition":
        from src.models.train_condition import make_crops, train_condition
        counts = make_crops(Path(args.manifest), Path(args.raw_classes), Path(args.crops), args.per_class)
        metrics = train_condition(Path(args.crops), Path(args.checkpoint), args.epochs, args.batch)
        print(json.dumps({"crop_counts": counts, "metrics": metrics, "checkpoint": args.checkpoint}, indent=2))



if __name__ == "__main__":
    main()
