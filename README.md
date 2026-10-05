# Trustworthy AI-Assisted Coral Reef Health Assessment

A reproducible image/video workflow for visible benthic composition and coral-condition indicators. Every result carries image-quality checks, predicted coverage, confidence/entropy signals, and an explicit `accepted` or `needs_review` state.

Scope: broad visible benthic/condition **indications**, not an ecosystem-health diagnosis, not a Lakshadweep validation.

---

## Table of contents

1. [Current artifacts](#current-artifacts)
2. [Dataset](#dataset)
3. [Quick start](#quick-start)
4. [Entry point: `main.py`](#entry-point-mainpy)
5. [Data preparation](#data-preparation)
6. [Training (Kaggle kernel)](#training-kaggle-kernel)
7. [Condition classifier](#condition-classifier)
8. [Evaluation](#evaluation)
9. [Inference pipeline](#inference-pipeline)
10. [Trust / reliability gate](#trust--reliability-gate)
11. [Overlay and video aggregation](#overlay-and-video-aggregation)
12. [Streamlit UI](#streamlit-ui)
13. [Repository layout](#repository-layout)
14. [Reproducibility record](#reproducibility-record)

---

## Current artifacts

- **Active segmentation checkpoint**: `models/checkpoints/yolo26n_seg/weights/best.pt`
  - Kaggle T4, 30 epochs, imgsz 512, batch 16, full Coralscapes training set (1,517 train images)
  - Mask mAP50 ≈ 0.165 on the val split (from `models/checkpoints/yolo26n_seg/results.csv`)
  - Operating IoU ≈ 0.259 at conf 0.20 (20-image operating report), full-set mIoU ≈ 0.392 (all 392 test images)
- **Condition classifier**: `models/checkpoints/condition_resnet18.pt`
  - ResNet18, trained on crops split 250/class (live/bleached/dead) from segmentation masks
- **Reports** (JSON): `reports/segmentation_metrics.json` (operating threshold 0.20, 20 images), `reports/segmentation_metrics_diagnostic.json` (conf 0.001 recall diagnostic), `reports/segmentation_metrics_full_v2.json` (full 392-image test)
- **Training record**: `reports/training_run.json`
- **Prepared dataset**: `data/processed/coralscapes_yolo/manifest.csv` (2,075 images; train/val/test = 1,517/166/392)
- **Sample inference**: `outputs/final_sample/overlay.jpg`, `outputs/final_video_inference/`
- **Original Kaggle run assets** are gone from the tree; only the selected best.pt/last.pt outputs are kept in `models/checkpoints/`.

## Dataset

- Source: `EPFL-ECEO/coralscapes` (Coralscapes, ICCVW 2025)
  - Local snapshot: `datasets/coralscapes_hf/` (5.86 GB, parquet)
  - Original Zenodo record 15061505 is archived only for provenance; the Zenodo tree is no longer required
- 39 raw classes mapped to 7 broad visible classes (`coral_alive`, `coral_bleached`, `coral_dead`, `algae`, `rubble`, `sand_rock`, `other`) — logic in `src/data/label_map.py`

## Quick start

```bash
python -m pip install -r requirements.txt

# 1. Prepare YOLO-format dataset from the local HF snapshot
python main.py prepare

# 2. Full training is the Kaggle kernel (see "Training (Kaggle kernel)").
#    The trained best.pt/last.pt and plots are already under models/.

# 3. The condition classifier checkpoint already exists at models/checkpoints/condition_resnet18.pt.

# 4. Evaluate segmentation
python main.py evaluate --model models/checkpoints/yolo26n_seg/weights/best.pt --max-images 20

# 5. Single image inference
python main.py infer --model models/checkpoints/yolo26n_seg/weights/best.pt --image path/to/image.jpg

# 6. Video inference
python main.py infer --model models/checkpoints/yolo26n_seg/weights/best.pt --video path/to/video.mp4 --out outputs/video_analysis

# 7. Streamlit UI
python main.py app
```

## Entry point: `main.py`

Every command-line workflow is funneled through `main.py`:

```bash
python main.py prepare [--root DATASETS_HF] [--out PROCESSED] [--max-images-per-split N] [--copy-images]
python main.py evaluate --model CKPT [--manifest ...] [--raw-classes ...] [--max-images N] [--conf ...] [--out ...]
python main.py infer --model CKPT [--image IMG | --video VID] [--out DIR] [--imgsz 512] [--conf ...]
python main.py app [--port 8501]
```

- `prepare` calls `src/data/prepare.py:prepare()` — converts the local HF Coralscapes snapshot into the YOLO segmentation dataset (manifest.csv, data.yaml, label_map.json, classes.json, YOLO polygon labels).
- `evaluate` calls `src/evaluate/segmentation.py:evaluate()` — per-class IoU, mIoU, confusion matrix.
- `infer` calls `src/inference/pipeline.py:CoralPipeline` for image or video, writes overlay + JSON report.
- `app` launches `src/app/streamlit_app.py` as a Streamlit sub-process.

Training itself no longer runs in a local script; it is the Kaggle kernel defined in `kaggle_run/`. Nothing under `scripts/` remains.

## Data preparation

Entry point: `python main.py prepare` → `src/data/prepare.py:prepare()`

Two source modes, chosen by `--root`:

- **HF snapshot (default, `datasets/coralscapes_hf`)**: `prepare_from_hf_snapshot()` reads `id2label.json`, classifies the split via `data/{train,validation,test}-*.parquet` (pyarrow direct, no HF cache copy). Each row: `image` decode to PNG, `label` decode to raw mask PNG.
  - For each sample it writes:
    - `data/processed/coralscapes_yolo/images/<split>/<stem>.png` (RGB PNG)
    - `data/processed/coralscapes_yolo/masks/<split>/<stem>.png` (raw label mask, full-res)
    - `data/processed/coralscapes_yolo/labels/<split>/<stem>.txt` (YOLO seg polygons, made by mapping mask → broad IDs → `cv2.findContours` + `approxPolyDP`; resized to 1024×512)
  - Emits `manifest.csv` (image, mask, split, site, group, label_count), `data.yaml` for Ultralytics, `label_map.json` (raw class→broad ID map), `classes.json` (broad class list for training_condition). Counts written at the end.
- **Old layout (Zenodo folder)**: `prepare()` expects the same `root/leftImg8bit/...` + `gtFine/...` structure; uses symlink (default) or `--copy-images`. This path is for the old Zenodo tree only.

The symlink path (`os.symlink`) is used because 5.5 GB×2 was impractical on disk; Kaggle/HF path copies files because the kernel's own tmpdir has no symlink source.

Label conversion: `raw_to_broad` maps each raw class to one of the 7 broad IDs; 255 = ignored. Then for each broad id, contours are converted to normalized polygon coordinates, one line per polygon.

## Training (Kaggle kernel)

`kaggle_run/train_kaggle.py` + `kernel-metadata.json` define the remote job. The kernel is script-type, `enable_gpu: true`, `enable_internet: true`, accelerator `NvidiaTeslaT4`; kernel title `nahilr/coral-btp-train`.

Script flow:

1. `pip install ultralytics datasets huggingface_hub` (Kaggle image already has torch/torchvision/opencv)
2. Load `EPFL-ECEO/coralscapes` streaming from the Hub, iterate each split, replicate local conversion inline: map with the same 7-class rule, contours→polygons, save to `/kaggle/working/coralscapes_yolo_hf/`
3. Train `YOLO('yolo26n-seg.pt')` with `epochs=30, imgsz=512, batch=16, device=0, workers=2, patience=10, plots=True`, output to `/kaggle/working/checkpoints/yolo26n_full/`
4. Log/plots/results saved under `/kaggle/working/checkpoints/yolo26n_full/`; `best.pt` — kept.

After the run:

```bash
kaggle kernels output nahilr/coral-btp-train -p kaggle_out_v2
```
(unzips the notebook output incl. `checkpoints/yolo26n_full/`)

## Condition classifier

Checkpoint: `models/checkpoints/condition_resnet18.pt` (ResNet18 trained on crops
sampled from the Coralscapes masks — live/bleached/dead coral, up to 250 crops per
class). The crop-extraction + training code path was local and is no longer part of
the working tree; the checkpoint is kept as a standalone artifact for any future
re-training.

## Evaluation

Entry point: `python main.py evaluate` → `src/evaluate/segmentation.py:evaluate()`

For each row in `manifest.csv` (filtered to `split == "test"`, optionally truncated by `--max-images N`):

1. Load the image with cv2 and the raw mask via `broad_mask()` to get broad ground-truth class IDs.
2. Predict with Ultralytics at `imgsz=384` now (image resized internally; we trained at 512 but evaluate at 384 historical; keep accordance with the *training* imgsz of the active model).
3. Produce an 8-class confusion map: pixels with no predicted mask are treated as `other` (last broad class), to avoid silently丢弃 negative recall.
4. Per-class IoU = TP / union, mIoU = mean of the 7.

Emitted JSON: `{images, confidence_threshold, confusion_matrix, per_class_iou, mIoU}`.

Two operating reports are kept:
- Operating: `conf 0.20` (conservative acceptance policy) — mIoU 0.259 @ 20 images.
- Diagnostic: `conf 0.001` — diagnostic for recall inspection, not for acceptance — mIoU 0.243.
- Full-set: `segmentation_metrics_full_v2.json` against the same Kaggle checkpoint on all 392 test images — mIoU 0.392.

## Inference pipeline

`src/inference/pipeline.py` (`CoralPipeline`):

- `__init__(model_path, imgsz=384, conf=0.20)`: holds a YOLO model and predictions.
- `infer(image_path, save_overlay=None)`:
  - `assess_image()` gives the quality dict (see below).
  - `model.predict` at the chosen imgsz/conf → masks, boxes, classes, scores.
  - Build class map per pixel (highest-conf mask wins replacement) and confidence map.
  - Per-class pixel shares `proportions`, `coverage = masked pixels / total`, `mean_confidence`, normalized `entropy` over the 7-class distribution.
  - `review_decision()` decides `accepted` vs `needs_review` (see below).
  - If `save_overlay` is set, draws the annotated overlay (see next section).
  - Returns a dict with model path, image info, quality, coverage, mean_conf, entropy, composition, detections, reliability, overlay.
- `infer_video()`: walks an input video, samples one frame every `every_n` frames up to `max_frames`; each sampled frame goes through `infer()` and overlay is saved; aggregated per-video summary with framewise reliability; rejected frames (needs_review) are kept in the report with `accepted_frames = 0` semantics, never silently dropped.

## Trust / reliability gate

`src/trust/reliability.py`:

- `softmax` / `expected_calibration_error` / `normalized_entropy` — small utilities; ECE bincounts confidence-vs-correct.
- `review_decision(quality, confidence, coverage, entropy, thresholds)`:
  - copies any quality flags (blur, exposure, low_contrast, color_cast) into reasons
  - adds `low_confidence` if conf < 0.55, `low_predicted_coverage` if coverage < 0.20, `high_entropy` if entropy > 0.82
  - returns `{status: accepted|needs_review, reasons: sorted unique list, thresholds}`

Quality: `src/data/quality.py::assess_image` checks Laplacian-blur variance, brightness mean, contrast stddev, channel max/min color cast; scoring drops the quality score for each present flag and sets `needs_review` status when flags appear.

## Overlay and video aggregation

Overlay drawing is in `src/inference/pipeline.py` after mask blending:

1. **Mask blend**: each predicted mask is blended `0.55 * image + 0.45 * class_color`.
2. **Boxes + labels** (`_draw_annotations`): each instance box drawn with color by class; label text `class_name score` above each box. `fs` (font scale) = `max(0.5, min(1.6, width/900.0))`, `bt` (thickness) = `max(2, round(width/550))`.
3. **Top-left info panel**: coverage %, mean confidence, status, and per-class coverage percentages, on a semi-transparent dark panel width = `width * 0.32`.

Manual: the same overlay is reused for video by `infer_video()`. Per-frame overlays are `frame_NNNNNN_overlay.jpg` in the output dir; aggregate JSON has `sampled_frames`, framewise `frames`, and `summary` with mean composition and status across accepted frames (or all frames if none accepted).

## Streamlit UI

`src/app/streamlit_app.py`:

- Available image types: `jpg/jpeg/png`; video types: `mp4/avi/mov/mkv`.
- Two-pane width layout for images (input/overlay side-by-side, 500 px each).
- Video: `every_n` slider (5–60, default 15) and `max_frames` slider (4–60, default 12), spinner while running; shows `accepted_frames/total_frames` status banner, video-level summary JSON, up to 6 overlay frames (small grid), and the full per-frame report + download button.
- After `upload`: writes the bytes to a temp dir, invokes `CoralPipeline`, displays the overlay(s), saves the JSON report for `st.download_button`.

## Repository layout

```
BTP/
├── README.md
├── problem.md, sources.md, requirements.txt
├── configs/labels.yaml                    # class list alias
├── datasets/coralscapes_hf/               # HF snapshot (5.86 GB, parquet)
├── data/processed/coralscapes_yolo/       # manifest + YOLO labels + masks + data.yaml
├── data/processed/condition_crops/        # ResNet18 training crops
├── models/checkpoints/yolo26n_seg/        # active Kaggle T4 checkpoint
│   ├── weights/best.pt, weights/last.pt
│   ├── args.yaml, results.csv, results.png
│   ├── BoxF1_curve.png … confusion_matrix_normalized.png
├── models/checkpoints/condition_resnet18.pt
├── kaggle_run/                            # remote training job
│   ├── train_kaggle.py
│   └── kernel-metadata.json
├── main.py                             # single entrypoint (prepare/evaluate/infer/app)
├── src/
│   ├── data/label_map.py, prepare.py, quality.py
│   ├── evaluate/segmentation.py
│   ├── inference/pipeline.py
│   ├── trust/reliability.py
│   └── app/streamlit_app.py
├── reports/
│   ├── segmentation_metrics.json
│   ├── segmentation_metrics_diagnostic.json
│   ├── segmentation_metrics_full_v2.json
│   └── training_run.json
├── outputs/
│   ├── final_sample/overlay.jpg, report.json
│   └── final_video_inference/{frame_000000.jpg, frame_000000_overlay.jpg, report.json}
└── project/ , papers/ , archived dirs
```

## Reproducibility record

`reports/training_run.json` now documents:

```json
{
  "project": "Trustworthy AI-Assisted Coral Reef Health Assessment",
  "dataset": {"name":"Coralscapes","prepared_manifest":"...","images":2075,"by_split":{...},"classes":"configs/labels.yaml"},
  "segmentation": {"architecture":"Ultralytics YOLO26n-seg","checkpoint":"models/checkpoints/yolo26n_seg/weights/best.pt","trainer":"Kaggle T4","epochs":30,"imgsz":512,"batch":16,"operating_confidence":0.20,"metrics_mAP50_mask":0.1652},
  ...
  "evaluation_reports":["reports/segmentation_metrics.json","reports/segmentation_metrics_diagnostic.json","reports/segmentation_metrics_full_v2.json"]
}
```

Reproduce from scratch a smaller slice:

```bash
python main.py prepare --root datasets/coralscapes_hf --out /tmp/prep --max-images-per-split 200
# train a local smoke run (small epochs)
# then kaggle kernels version:            kaggle_run/train_kaggle.py
```

---

**Notice**: `archive` of old Zenodo datasets had 9–13 GB of GBIF/Dryad/NOAA files; they were removed along with the other-source assets. If you need them back, see `project/dataset-inventory.md` links and the earlier-agreed-upon policy of re-downloading on demand.

## Assumptions

- HF snapshot is the single dataset source for both local prep and Kaggle.
- The Kaggle T4 checkpoint at `models/checkpoints/yolo26n_seg/weights/best.pt` is the production artifact.
- Kaggle full model is the production artifact; local training entrypoints are removed because training always happens via Kaggle.
