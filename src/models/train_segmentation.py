from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


def train(data: Path, output: Path, epochs=1, imgsz=512, batch=2, device=None, fraction=1.0):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = device or (0 if torch.cuda.is_available() else "cpu")
    model = YOLO("yolo26n-seg.pt")
    results = model.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch,
        device=device, project=str(output), name="yolo26n_seg", exist_ok=True,
        workers=2, pretrained=True, cache=False, plots=True,
        patience=max(2, epochs), fraction=fraction, seed=42, verbose=True)
    best = Path(results.save_dir) / "weights" / "best.pt"
    return {"save_dir": str(results.save_dir), "best": str(best), "device": str(device),
            "epochs": epochs, "fraction": fraction}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/processed/coralscapes_yolo/data.yaml"))
    p.add_argument("--output", type=Path, default=Path("models/checkpoints"))
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    print(json.dumps(train(args.data, args.output, args.epochs, args.imgsz, args.batch, args.device, args.fraction), indent=2))


if __name__ == "__main__":
    main()
