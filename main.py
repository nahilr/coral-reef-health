#!/usr/bin/env python3
"""Single entry point for the coral-reef trust pipeline.

Subcommands:
  prepare          prepare the YOLO seg dataset from the local HF snapshot
  evaluate         evaluate a segmentation checkpoint on the held-out test split
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
    prep.add_argument("--root", type=Path, default=Path("datasets/coralscapes_hf"))
    prep.add_argument("--out", type=Path, default=Path("data/processed/coralscapes_yolo"))
    prep.add_argument("--max-images-per-split", type=int, default=None)
    prep.add_argument("--copy-images", action="store_true")

    ev = sub.add_parser("evaluate", help="evaluate segmentation checkpoint")
    ev.add_argument("--model", default="models/checkpoints/yolo26n_seg/weights/best.pt")
    ev.add_argument("--manifest", default="data/processed/coralscapes_yolo/manifest.csv")
    ev.add_argument("--raw-classes", default="data/processed/coralscapes_yolo/classes.json")
    ev.add_argument("--max-images", type=int, default=None)
    ev.add_argument("--conf", type=float, default=0.20)
    ev.add_argument("--out", default="reports/segmentation_metrics.json")

    inf = sub.add_parser("infer", help="image or video inference")
    inf.add_argument("--model", default="models/checkpoints/yolo26n_seg/weights/best.pt")
    inf.add_argument("--condition-model", default="models/checkpoints/condition_resnet18.pt")
    inf.add_argument("--image")
    inf.add_argument("--video")
    inf.add_argument("--out", default="outputs/inference")
    inf.add_argument("--imgsz", type=int, default=512)
    inf.add_argument("--conf", type=float, default=0.20)

    app = sub.add_parser("app", help="start the Streamlit UI")
    app.add_argument("--port", type=int, default=8501)

    tc = sub.add_parser("train-condition", help="extract coral crops and train condition classifier")
    tc.add_argument("--manifest", default="data/processed/coralscapes_yolo/manifest.csv")
    tc.add_argument("--raw-classes", default="data/processed/coralscapes_yolo/classes.json")
    tc.add_argument("--crops", default="data/processed/condition_crops")
    tc.add_argument("--checkpoint", default="models/checkpoints/condition_resnet18.pt")
    tc.add_argument("--per-class", type=int, default=250)
    tc.add_argument("--epochs", type=int, default=1)
    tc.add_argument("--batch", type=int, default=32)

    args = p.parse_args()

    if args.cmd == "prepare":
        from src.data.prepare import prepare
        print(json.dumps(prepare(args.root, args.out, args.max_images_per_split, args.copy_images), indent=2))
    elif args.cmd == "evaluate":
        from src.evaluate.segmentation import evaluate
        print(json.dumps(evaluate(args.model, args.manifest, args.raw_classes,
                                  max_images=args.max_images, conf=args.conf, out=args.out), indent=2))
    elif args.cmd == "infer":
        if not args.image and not args.video:
            p.error("Must specify either --image or --video")
        from src.inference.pipeline import CoralPipeline
        pipeline = CoralPipeline(args.model, condition_model_path=args.condition_model,
                                 imgsz=args.imgsz, conf=args.conf)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        if args.image:
            report = pipeline.infer(args.image, out / "overlay.jpg")
            (out / "report.json").write_text(json.dumps(report, indent=2))
        else:
            report = pipeline.infer_video(args.video, args.out)
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
