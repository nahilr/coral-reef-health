from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference.pipeline import CoralPipeline

p = argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--image"); p.add_argument("--video"); p.add_argument("--out", default="outputs/inference"); args = p.parse_args()
pipeline = CoralPipeline(args.model)
if args.image:
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); report = pipeline.infer(args.image, out / "overlay.jpg"); (out / "report.json").write_text(json.dumps(report, indent=2))
else:
    report = pipeline.infer_video(args.video, args.out)
print(json.dumps(report, indent=2))
