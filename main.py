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
    inf.add_argument("--dir", dest="input_dir",
                     help="Infer supported images/videos from this input folder recursively")
    inf.add_argument("--out", default="outputs/inference",
                     help="Output folder for annotated media, JSON reports, and batch manifest")
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
        provided_inputs = sum(bool(value) for value in (args.image, args.video, args.input_dir))
        if provided_inputs != 1:
            p.error("Specify exactly one of --image, --video, or --dir")
        from src.inference.pipeline import CoralPipeline
        pipeline = CoralPipeline(args.model, imgsz=args.imgsz, conf=args.conf, model_type=args.model_type)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        if args.input_dir:
            input_dir = Path(args.input_dir)
            if not input_dir.is_dir():
                p.error(f"Input folder does not exist or is not a directory: {input_dir}")
            image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            video_exts = {".mp4", ".avi", ".mov", ".mkv"}
            files = sorted(
                path for path in input_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in image_exts | video_exts
            )
            if not files:
                p.error(f"No supported image or video files found under: {input_dir}")

            entries = []
            for index, source in enumerate(files, start=1):
                relative = source.relative_to(input_dir)
                kind = "images" if source.suffix.lower() in image_exts else "videos"
                overlay_dir = out / relative.parent
                overlay_dir.mkdir(parents=True, exist_ok=True)
                report_dir = out / relative.parent
                report_dir.mkdir(parents=True, exist_ok=True)
                report_path = report_dir / f"{source.name}.json"
                print(f"[{index}/{len(files)}] {kind[:-1]}: {relative}", flush=True)
                try:
                    if kind == "images":
                        overlay_path = overlay_dir / f"{source.name}.overlay.jpg"
                        report = pipeline.infer(source, overlay_path)
                        report_path.write_text(json.dumps(report, indent=2))
                        entry = {
                            "input": str(source),
                            "type": "image",
                            "overlay": str(overlay_path),
                            "report": str(report_path),
                            "status": "complete",
                        }
                    else:
                        video_name = f"{source.stem}_annotated.mp4"
                        report = pipeline.infer_video(
                            source, overlay_dir,
                            output_video_name=video_name,
                            every_n=args.every_n,
                            max_frames=args.max_frames,
                            save_frames=args.save_frames,
                            report_path=report_path,
                        )
                        entry = {
                            "input": str(source),
                            "type": "video",
                            "annotated_video": report.get("annotated_video"),
                            "report": str(report_path),
                            "processed_frames": report.get("sampled_frames", 0),
                            "status": "complete",
                        }
                    entries.append(entry)
                    print(f"    complete: {entry.get('overlay') or entry.get('annotated_video')}", flush=True)
                except Exception as exc:
                    entry = {
                        "input": str(source),
                        "type": "image" if kind == "images" else "video",
                        "status": "error",
                        "error": str(exc),
                    }
                    entries.append(entry)
                    print(f"    ERROR: {exc}", file=sys.stderr, flush=True)

            manifest = {
                "input_dir": str(input_dir),
                "output_dir": str(out),
                "model": args.model,
                "model_type": args.model_type,
                "imgsz": args.imgsz,
                "conf": args.conf,
                "every_n": args.every_n,
                "max_frames": args.max_frames,
                "processed": sum(entry["status"] == "complete" for entry in entries),
                "failed": sum(entry["status"] == "error" for entry in entries),
                "files": entries,
            }
            manifest_path = out / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(f"Processed {manifest['processed']} of {len(entries)} files; {manifest['failed']} failed.")
            print(f"Batch manifest: {manifest_path}")
        elif args.image:
            report = pipeline.infer(args.image, out / "overlay.jpg")
            (out / "report.json").write_text(json.dumps(report, indent=2))
        else:
            report = pipeline.infer_video(args.video, args.out,
                                          output_video_name=args.output_video_name,
                                          every_n=args.every_n,
                                          max_frames=args.max_frames,
                                          save_frames=args.save_frames,
                                          report_path=out / "report.json")
            if report.get("annotated_video"):
                print(f"Annotated video saved to: {report['annotated_video']}")
        if not args.input_dir:
            print(json.dumps(report, indent=2))
    elif args.cmd == "app":
        cmd = [sys.executable, "-m", "streamlit", "run", "src/app/streamlit_app.py", "--server.port", str(args.port)]
        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
