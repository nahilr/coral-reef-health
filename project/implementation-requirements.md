# Implementation Requirements

## Recommended environment

| Need | Local machine | Google Colab |
|---|---|---|
| Best use | Data preparation, inference, demo, source control | Training and experiments |
| GPU | NVIDIA CUDA GPU; 8 GB VRAM for nano models, 12-16 GB preferred | Free/paid runtime GPU availability varies; a T4-class GPU is enough for nano/small 512-640 px runs |
| CPU/RAM | 4+ cores and 16 GB RAM recommended | Provided by runtime |
| Storage | 50-100 GB free SSD recommended | Mount Drive or copy only the active dataset/checkpoint |
| OS | Linux/Windows/macOS; CUDA path easiest on Linux/Windows NVIDIA | Browser plus Google account |

## Software

- Python 3.10 or 3.11.
- PyTorch matched to the selected CUDA runtime.
- Ultralytics (YOLO26), OpenCV, Albumentations, scikit-learn, pandas, NumPy, Matplotlib/Seaborn.
- FiftyOne or CVAT/Label Studio only if manual annotation/review is needed.
- Streamlit or Gradio for the local demo.
- Git, a `requirements.txt`/lock file, fixed random seeds, and experiment logs.

## Repository layout

```text
data/raw/                 # never edited
data/interim/             # converted masks/crops
data/manifests/           # labels, splits, data cards
models/checkpoints/       # excluded from Git if large
src/data/                 # conversion and validation
src/train/                # model configurations
src/evaluate/             # metrics, calibration, OOD
src/app/                  # upload/report demo
reports/figures/          # final charts and failure gallery
```

## Colab operating rules

1. Keep the notebook small: setup, mount/copy data, one config, train, evaluate, export. Put reusable logic in versioned Python files.
2. Save checkpoints, split manifests, metrics and the exact configuration to Drive after each run.
3. Pin package versions and write the GPU/runtime details into every experiment record.
4. Start with `yolo26n-sem` at 512 px and a batch size that fits; scale only after a successful end-to-end run.
5. Never use the final test set while iterating in Colab.

## Local-only feasibility

- **No GPU:** data inspection, annotations, conversion, demo inference with a nano exported model, and report generation are feasible; model training should use Colab.
- **8 GB NVIDIA GPU:** core segmentation/classification training is feasible with nano models, mixed precision, smaller batches and 512 px images.
- **12-16 GB NVIDIA GPU:** recommended balance for repeatable experimentation at 640 px and a small condition model.
- **Apple Silicon / AMD:** inference may work through supported back ends, but make Colab the training path unless the PyTorch accelerator setup is already verified.

## Acceptance checklist

- Dataset card and licence recorded for every source.
- Grouped split manifest immutable before training.
- Model metrics and calibration reported on validation and sealed test data.
- At least 20 qualitatively reviewed successes and 20 failures, including low-quality/OOD inputs.
- Demo never hides rejected inputs; it labels them `needs review`.
