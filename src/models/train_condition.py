from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from src.data.label_map import broad_mask, load_coralscapes_classes, make_raw_to_broad

CONDITION_CLASSES = ["coral_alive", "coral_bleached", "coral_dead"]


def make_crops(manifest: Path, raw_classes: Path, out: Path, per_class: int = 250, seed: int = 42):
    rng = np.random.default_rng(seed)
    df = pd.read_csv(manifest)
    mapping = make_raw_to_broad(load_coralscapes_classes(raw_classes))
    for split in ("train", "val"):
        for name in CONDITION_CLASSES:
            (out / split / name).mkdir(parents=True, exist_ok=True)
    counts = {"train": {c: 0 for c in CONDITION_CLASSES}, "val": {c: 0 for c in CONDITION_CLASSES}}
    for row in df.itertuples():
        if row.split not in counts:
            continue
        image = cv2.imread(row.image)
        raw = cv2.imread(row.mask, cv2.IMREAD_UNCHANGED)
        mapped = broad_mask(raw, mapping)
        for class_id, name in enumerate(CONDITION_CLASSES):
            if counts[row.split][name] >= per_class:
                continue
            ys, xs = np.where(mapped == class_id)
            if len(xs) == 0:
                continue
            take = min(2, per_class - counts[row.split][name], len(xs))
            idx = rng.choice(len(xs), size=take, replace=False)
            for j in idx:
                x, y = int(xs[j]), int(ys[j])
                half = 64
                crop = image[max(0, y-half):min(image.shape[0], y+half), max(0, x-half):min(image.shape[1], x+half)]
                if crop.shape[0] < 24 or crop.shape[1] < 24:
                    continue
                path = out / row.split / name / f"{Path(row.image).stem}_{x}_{y}.jpg"
                cv2.imwrite(str(path), crop)
                counts[row.split][name] += 1
    return counts


def train_condition(crop_root: Path, output: Path, epochs: int = 1, batch: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf = transforms.Compose([transforms.Resize((128, 128)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    eval_tf = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_set = datasets.ImageFolder(crop_root / "train", transform=train_tf)
    val_set = datasets.ImageFolder(crop_root / "val", transform=eval_tf)
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch, shuffle=False, num_workers=2)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(train_set.classes))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss_fn(model(x), y).backward(); opt.step()
    model.eval(); truth, pred = [], []
    with torch.no_grad():
        for x, y in val_loader:
            pred.extend(model(x.to(device)).argmax(1).cpu().tolist()); truth.extend(y.tolist())
    metrics = {"classes": train_set.classes, "train_images": len(train_set), "val_images": len(val_set), "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)) if truth else 0.0, "balanced_accuracy": float(balanced_accuracy_score(truth, pred)) if truth else 0.0, "classification_report": classification_report(truth, pred, target_names=train_set.classes, output_dict=True, zero_division=0) if truth else {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": train_set.classes, "metrics": metrics}, output)
    return metrics


def main():
    p = argparse.ArgumentParser(); p.add_argument("--manifest", default="data/processed/coralscapes_yolo/manifest.csv"); p.add_argument("--raw-classes", default="datasets/coralscapes/classes.json"); p.add_argument("--crops", default="data/processed/condition_crops"); p.add_argument("--checkpoint", default="models/checkpoints/condition_resnet18.pt"); p.add_argument("--per-class", type=int, default=250); p.add_argument("--epochs", type=int, default=1); p.add_argument("--batch", type=int, default=32)
    a = p.parse_args(); counts = make_crops(Path(a.manifest), Path(a.raw_classes), Path(a.crops), a.per_class); metrics = train_condition(Path(a.crops), Path(a.checkpoint), a.epochs, a.batch); print(json.dumps({"crop_counts": counts, "metrics": metrics, "checkpoint": a.checkpoint}, indent=2))


if __name__ == "__main__": main()
