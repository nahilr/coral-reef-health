# Data

All dataset, extracted, processed, and evaluation directories are git-ignored; this README is tracked.

## Directory Structure

| Directory | Purpose | Retention |
|---|---|---|
| `raw/` | Original dataset downloads, archives, and reference survey media | Immutable source data. Keep unchanged. |
| `extracted/` | Unpacked image splits and standardized video clips | Rebuildable from `raw/`. |
| `eval/` | Frozen test fixtures and cached model prediction maps | Keep test split frozen; predictions disposable. |
| `processed/` | Model inference outputs, diagnostic overlays, and prepared YOLO data | Rebuildable / disposable. |

```text
data/
├── raw/                      # Source datasets, Parquet shards, and survey videos
│   ├── coralscapes_v2/       # CoralscapesV2 Parquet splits and class JSONs
│   ├── video_10624794/       # DeepReefMap underwater survey videos
│   ├── lter_mcr/             # Moorea Coral Reef (MCR) LTER survey data
│   └── 0269246.1.1/          # NOAA ESD Coral Bleaching dataset
├── extracted/                # Extracted frames and standardized test clips
│   ├── coralscapesv2/        # Extracted train/val/test image directories
│   └── clips/                # Sampled 20-second underwater video clips
├── eval/                     # Fixed benchmark fixtures and cached predictions
│   └── coralscapesv2-test/   # 473-image test split and prediction cache
└── processed/                # Evaluation run artifacts and prepared training data
    ├── yolo26s_sem_test/     # Overlays, class masks, and JSON reports
    └── yolo/                 # 7-class YOLO segmentation format (rebuildable)
```

## Workflows

### 1. Prepare YOLO training data
Generate the 7-class YOLO segmentation format from CoralscapesV2 Parquet shards:

```bash
python main.py prepare --root data/raw/coralscapes_v2 --out data/processed/yolo
```

### 2. Run multi-backend benchmark
Evaluate models on the frozen test fixture:

```bash
python scripts/evaluate_model_comparison.py --models yolo,yolo_s
```

### 3. Evaluate a single checkpoint
```bash
python main.py evaluate \
  --model models/checkpoints/yolo26n_sem/weights/best.pt \
  --manifest data/processed/yolo/manifest.csv \
  --raw-classes data/raw/coralscapes_v2/id2label.json \
  --out reports/eval_metrics.json
```
