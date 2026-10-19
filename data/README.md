# Data

All data is git-ignored except this README. The folder is organised into tiers:

| Tier | Purpose | Rebuildable? |
|---|---|---|
| `raw/` | Immutable downloads (HF snapshots, external datasets) | Re-download with pinned revision |
| `interim/` | Scratch conversions | Yes, delete freely |
| `processed/` | Model-ready datasets (YOLO format) | Yes, via `main.py prepare` / `prepare-v2` |
| `eval/` | Fixed benchmark fixtures (images, masks, predictions) | Frozen; version before comparing |
| `cache/` | HF/torch download cache | Yes |

## Contents

- `raw/coralscapes_hf/` — V1 Coralscapes HF snapshot (Parquet)
- `raw/coralscapes_v2/` — V2 CoralscapesV2 HF snapshot (optional, ~6.75 GB)
- `raw/lter_mcr/` — MCR-LTER labeled corals dataset
- `raw/video_10624794/` — ROV/video dataset (DeepReefMap-style)
- `raw/clips/` — Loose video clips for inference testing
- `processed/v1_yolo/` — V1 7-class YOLO segmentation dataset
- `processed/v1_condition_crops/` — V1 condition classifier crops
- `processed/v2_yolo/` — V2 20-class YOLO dataset (built by `prepare-v2`)
- `eval/coralscapesv2-test/` — 473-image official test split + id2label
- `eval/predictions/` — Cached prediction maps keyed by model version

## Rebuild commands

```bash
# V1 dataset
python main.py prepare --root data/raw/coralscapes_hf --out data/processed/v1_yolo

# V2 dataset
python main.py prepare-v2 --root data/raw/coralscapes_v2 --out data/processed/v2_yolo

# V1 evaluation
python main.py evaluate --manifest data/processed/v1_yolo/manifest.csv \
  --raw-classes data/processed/v1_yolo/classes.json

# V2 evaluation
python main.py evaluate-v2 --manifest data/processed/v2_yolo/manifest.csv \
  --raw-classes data/raw/coralscapes_v2/id2label.json
```

## Retention

- `raw/` is the only irreplaceable tier — keep pinned revisions.
- `processed/` and `cache/` are disposable; delete to reclaim space.
- `eval/` is frozen; never regenerate in place during a model comparison.
- `predictions/` is a cache keyed by `<model-version>`; safe to delete.
