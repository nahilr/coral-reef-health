# Trustworthy AI-Assisted Coral Reef Health Assessment

This repository contains a reproducible image/video workflow for visible benthic composition and coral-condition indicators. Every result includes image-quality checks, predicted coverage, confidence/entropy signals, and an explicit `accepted` or `needs_review` state.

## Quick start

```bash
python -m pip install -r requirements.txt
python scripts/prepare_dataset.py
# Trained on Kaggle via kaggle_run/train_kaggle.py (30 epochs on T4)
python scripts/evaluate.py --model models/checkpoints/yolo26n_seg/weights/best.pt --max-images 20
python scripts/infer.py --model models/checkpoints/yolo26n_seg/weights/best.pt --image path/to/image.jpg
streamlit run src/app/streamlit_app.py
```

The default preparation uses local Coralscapes images and masks and maps the original 39 classes to seven visible classes: live-looking coral, bleached-looking coral, dead coral, algae, rubble, sand/rock, and other. Raw labels remain in `data/processed/coralscapes_yolo/label_map.json`.

Checkpoints and plots are saved under `models/checkpoints/`; reports go under `reports/` and inference outputs under `outputs/`. The default preparation uses local symlinks for speed; use `python scripts/prepare_dataset.py --copy-images` when moving the processed dataset to Kaggle/Drive. Training itself is a Kaggle job (`kaggle_run/`), not a local loop.

The evaluator defaults to the conservative operating threshold (`--conf 0.20`). For a diagnostic recall check, run a second report with `--conf 0.001`; keep both reports because a low threshold is not an acceptance policy.

The system reports visible image indicators. It does not diagnose total ecosystem health, species, disease, or causal environmental effects. Models trained on Red Sea or other external imagery are source-transfer experiments until local Lakshadweep imagery has been independently labelled and evaluated.

## Current local artifacts

- Segmentation checkpoint: `models/checkpoints/yolo26n_seg/weights/best.pt`
- Condition crop classifier: `models/checkpoints/condition_resnet18.pt`
- Prepared Coralscapes manifest: `data/processed/coralscapes_yolo/manifest.csv` (2,075 images; train/val/test = 1,517/166/392)
- Operating evaluation: `reports/segmentation_metrics.json`
- Low-threshold diagnostic evaluation: `reports/segmentation_metrics_diagnostic.json`
- Reproducibility record: `reports/training_run.json`
- Image and video examples: `outputs/final_sample/` and `outputs/final_video_inference/`

The active checkpoint (`models/checkpoints/yolo26n_seg/weights/best.pt`) is the Kaggle T4 model (`yolo26n_full`, 30 epochs at 512px on the full dataset, mAP50(M) 0.165). The reliability gate abstains when confidence, coverage, entropy, or image quality is insufficient; the current metrics should be treated as a baseline, not as a field deployment result.
