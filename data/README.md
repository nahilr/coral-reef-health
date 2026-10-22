# Data

All dataset, processed, and evaluation files are git-ignored; this README is tracked.

| Folder | Purpose | Retention |
|---|---|---|
| `raw/` | Original dataset downloads and demo media | Keep source data unchanged; record revision and licence. |
| `processed/` | 7-class YOLO data generated from CoralscapesV2 | Rebuildable with `main.py prepare`. |
| `eval/coralscapesv2-test/` | Fixed 473-image benchmark fixture used by the five-backend evaluator | Treat as frozen during comparisons. |
| `eval/predictions/` | Cached prediction maps keyed by model and taxonomy | Disposable; safe to remove and regenerate. |

## Current data

- `raw/coralscapes_hf/` contains the original Coralscapes dataset. It is a historical local snapshot and does not use the 95-class CoralscapesV2 mapping expected by the current preparation workflow.
- `raw/coralscapes_v2/` is the expected local download location for the CoralscapesV2 Parquet snapshot. The dataset is not currently downloaded there; obtain it from `josauder/coralscapesV2` before running local preparation.
- `eval/coralscapesv2-test/` contains the image/mask pairs and `id2label.json` required by `scripts/evaluate_model_comparison.py`.
- `raw/lter_mcr/`, `raw/video_10624794/`, and `raw/clips/` contain supporting reference data or demo media; they are not inputs to the current model training/evaluation workflow.

## Rebuild and evaluate

The preparation command expects `id2label.json` at the snapshot root and Parquet shards under `data/`, with `train-*`, `validation-*`, and `test-*` splits.

```bash
python main.py prepare --root data/raw/coralscapes_v2 --out data/processed/yolo
```

Evaluate the prepared test split with the project taxonomy:

```bash
python main.py evaluate \
  --model models/checkpoints/yolo26n_sem/weights/best.pt \
  --manifest data/processed/yolo/manifest.csv \
  --raw-classes data/raw/coralscapes_v2/id2label.json \
  --out reports/eval_metrics.json
```

The reproducible five-backend benchmark uses the frozen fixture under `eval/coralscapesv2-test/` and is run with `scripts/evaluate_model_comparison.py`.

## Retention

- Keep raw inputs immutable and preserve their source/licence metadata.
- `processed/` and `eval/predictions/` can be rebuilt or deleted.
- Do not regenerate `eval/coralscapesv2-test/` in place during a model comparison.
