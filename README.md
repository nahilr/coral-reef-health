# Trustworthy AI-Assisted Coral Reef Health Assessment

A reproducible image/video workflow for visible benthic composition and coral-condition indicators. Every result carries image-quality checks, predicted coverage, confidence/uncertainty signals, and an explicit `accepted` or `needs_review` state.

Scope: quantitative visible benthic composition and coral-condition indicators with automated reliability verification.

---

## Table of contents

1. [Current artifacts](#current-artifacts)
2. [Dataset](#dataset)
3. [Quick start](#quick-start)
4. [Entry point: `main.py`](#entry-point-mainpy)
5. [Data preparation](#data-preparation)
6. [Training (Kaggle Cloud Workflow)](#training-kaggle-cloud-workflow)
7. [Condition classifier](#condition-classifier)
8. [Evaluation](#evaluation)
9. [Inference & reef health pipeline](#inference--reef-health-pipeline)
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
  - Operating mIoU ≈ 0.296 at conf 0.20 (20-image evaluation with Expected Calibration Error ECE = 0.208)
- **Condition classifier**: `models/checkpoints/condition_resnet18.pt`
  - ResNet18, evaluated on coral crops (live, bleached, dead) for secondary condition verification
- **Evaluation report**: `reports/segmentation_metrics.json` (per-class IoU, confusion matrix, and binned calibration stats)
- **Training record**: `reports/training_run.json`
- **Prepared dataset**: `data/processed/coralscapes_yolo/manifest.csv` (2,075 images; train/val/test = 1,517/166/392)
- **Sample inference**: `outputs/final_sample/overlay.jpg`, `outputs/final_sample/report.json`

## Dataset

- Source: `EPFL-ECEO/coralscapes` (Coralscapes, ICCVW 2025)
  - Canonical snapshot: `datasets/coralscapes_hf/` (5.86 GB, Parquet format)
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
python main.py train-condition [--manifest ...] [--raw-classes ...] [--crops ...] [--checkpoint ...] [--per-class N] [--epochs N] [--batch N]
python main.py app [--port 8501]
```

- `prepare` calls `src/data/prepare.py:prepare()` — converts the local HF Coralscapes snapshot into the YOLO segmentation dataset (manifest.csv, data.yaml, label_map.json, classes.json, YOLO polygon labels).
- `evaluate` calls `src/evaluate/segmentation.py:evaluate()` — per-class IoU, mIoU, confusion matrix.
- `infer` calls `src/inference/pipeline.py:CoralPipeline` for image or video, writes overlay + JSON report.
- `train-condition` calls `src/models/train_condition.py:make_crops()` + `train_condition()` — crop extraction + ResNet18 training; condition classifier checkpoint exists at `models/checkpoints/condition_resnet18.pt`.
- `app` launches `src/app/streamlit_app.py` as a Streamlit sub-process.

Training itself no longer runs in a local script; it is the Kaggle kernel defined in `kaggle_run/`. Nothing under `scripts/` remains.

## Data preparation

Entry point: `python main.py prepare` → `src/data/prepare.py:prepare()`

Reads `id2label.json` and split Parquet files directly from `datasets/coralscapes_hf/data/{train,validation,test}-*.parquet`. Each sample row decodes RGB image bytes and segmentation mask bytes:
- For each sample, it produces:
  - `data/processed/coralscapes_yolo/images/<split>/<stem>.png` (RGB image)
  - `data/processed/coralscapes_yolo/masks/<split>/<stem>.png` (Raw class mask)
  - `data/processed/coralscapes_yolo/labels/<split>/<stem>.txt` (Normalized YOLO segmentation polygons preserving native aspect ratio)
- Automatically emits:
  - `manifest.csv` (Sample index with paths, split, and polygon counts)
  - `data.yaml` (Ultralytics dataset configuration)
  - `label_map.json` (Raw 39-class to 7-broad-class dictionary)
  - `classes.json` (Class mapping for condition classification)

Label conversion: `raw_to_broad` collapses raw classes into 7 visible categories (`coral_alive`, `coral_bleached`, `coral_dead`, `algae`, `rubble`, `sand_rock`, `other`); background/dark pixels are assigned to 255 (ignored).

## Training (Kaggle Cloud Workflow)

### Architecture Partition: Kaggle vs. Local Machine

The project is strictly partitioned between **heavy cloud training** and **local deployment**:

```text
┌────────────────────────────────────────────────────────┐
│                   KAGGLE (Cloud GPU)                   │
│                                                        │
│  • Task: Heavy segmentation training ONLY              │
│  • Hardware: Free Tesla T4 (16 GB VRAM)                │
│  • Files: kaggle_run/train_kaggle.py                   │
│           kaggle_run/kernel-metadata.json              │
│  • Output: best.pt, last.pt, training curves, CSV      │
└──────────────────────────┬─────────────────────────────┘
                           │ (Download weights once via CLI)
                           ▼
┌────────────────────────────────────────────────────────┐
│             LOCAL MACHINE (Your PC / Laptop)           │
│                                                        │
│  1. Data Preparation:       python main.py prepare     │
│  2. Model Evaluation:       python main.py evaluate    │
│  3. Image/Video Inference:  python main.py infer       │
│  4. Trust/Reliability Gate: src/trust/reliability.py   │
│  5. Interactive Dashboard:  python main.py app         │
│  6. Crop Classifier (Train):python main.py train-cond. │
└────────────────────────────────────────────────────────┘
```

> **Why train on Kaggle?**  
> Training YOLO26n-seg on 1,517 high-resolution images ($512\text{px}$) for 30 epochs takes **~1.5 hours** on a Kaggle Tesla T4 GPU (16 GB VRAM). Training locally on consumer 4 GB GPUs (like GTX 1650) or CPUs risks CUDA Out-Of-Memory errors and takes 20+ hours. Once `best.pt` is downloaded to `models/checkpoints/yolo26n_seg/weights/best.pt`, **no further Kaggle compute is needed**.

---

### Kaggle Codefiles & Configuration

1. **`kaggle_run/train_kaggle.py`**:
   The standalone, automated training script executed remotely inside the Kaggle environment.
   - **Stage 1 (Environment Setup)**: Installs runtime dependencies (`ultralytics`, `datasets`, `huggingface_hub`).
   - **Stage 2 (Label Mapping)**: Downloads `id2label.json` from `EPFL-ECEO/coralscapes` and maps raw classes to the canonical 7 visible benthic categories (`coral_alive`, `coral_bleached`, `coral_dead`, `algae`, `rubble`, `sand_rock`, `other`).
   - **Stage 3 (Streaming Data Conversion)**: Streams the Hugging Face dataset split-by-split (`train`, `validation`, `test`) without downloading bulky multi-gigabyte archives upfront. For each image, it converts segmentation masks into normalized YOLO polygon coordinates via `cv2.findContours` + `cv2.approxPolyDP` and saves them to `/kaggle/working/coralscapes_yolo_hf/`.
   - **Stage 4 (Dataset Configuration)**: Writes `/kaggle/working/coralscapes_yolo_hf/data.yaml` linking Ultralytics to the converted directories and class names.
   - **Stage 5 (Model Training)**: Loads base weights `yolo26n-seg.pt` and trains with mixed precision:
     ```python
     model.train(
         data="/kaggle/working/coralscapes_yolo_hf/data.yaml",
         epochs=30,
         imgsz=512,
         batch=16,
         device=0,
         workers=2,
         project="/kaggle/working/checkpoints",
         name="yolo26n_full",
         patience=10,
         plots=True,
     )
     ```
   - **Stage 6 (Export)**: Saves trained weights (`best.pt`, `last.pt`), training loss graphs, PR curves, and confusion matrix plots to `/kaggle/working/checkpoints/yolo26n_full/`.

2. **`kaggle_run/kernel-metadata.json`**:
   Configures the Kaggle CLI kernel runner:
   ```json
   {
     "id": "<your-kaggle-username>/coral-btp-train",
     "title": "coral-btp-train",
     "code_file": "train_kaggle.py",
     "language": "python",
     "kernel_type": "script",
     "is_private": "true",
     "enable_gpu": "true",
     "enable_internet": "true",
     "accelerator": "NvidiaTeslaT4"
   }
   ```

---

### Step-by-Step Kaggle Training Commands

Follow these steps if you want to retrain the segmentation model from scratch:

#### Step 1: Install and Authenticate Kaggle CLI
```bash
python -m pip install kaggle
```
Set up your Kaggle API credentials:
- Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → click **Create New Token** (downloads `kaggle.json`).
- Place it in `~/.kaggle/kaggle.json` and set secure permissions:
  ```bash
  mkdir -p ~/.kaggle
  cp /path/to/kaggle.json ~/.kaggle/
  chmod 600 ~/.kaggle/kaggle.json
  ```
- Verify authentication:
  ```bash
  kaggle kernels list --mine
  ```

#### Step 2: Configure Kernel Metadata
Open `kaggle_run/kernel-metadata.json` and update the `"id"` field with your Kaggle username:
```json
"id": "<your-kaggle-username>/coral-btp-train"
```

#### Step 3: Push the Script & Start Training
Submit the job to Kaggle's GPU cluster:
```bash
kaggle kernels push -p kaggle_run/
```
Kaggle will immediately queue and launch a headless script run on a Tesla T4 GPU.

#### Step 4: Monitor Training Progress & Logs
Check if the kernel is queued, running, or complete:
```bash
kaggle kernels status <your-kaggle-username>/coral-btp-train
```
Stream real-time terminal output and training progress:
```bash
kaggle kernels logs <your-kaggle-username>/coral-btp-train
```

#### Step 5: Download Trained Checkpoints & Outputs
Once the status shows `complete`, download the generated model weights and training reports:
```bash
# Download to a temporary results directory
kaggle kernels output <your-kaggle-username>/coral-btp-train -p kaggle_results/

# Copy the best weights into the active local checkpoint path
cp kaggle_results/checkpoints/yolo26n_full/weights/best.pt models/checkpoints/yolo26n_seg/weights/best.pt
```

#### Step 6: Seamless Local Handoff
Once `best.pt` is placed in `models/checkpoints/yolo26n_seg/weights/`, the entire local workflow runs immediately:
```bash
# 1. Evaluate held-out test split with calibration metrics
python main.py evaluate --model models/checkpoints/yolo26n_seg/weights/best.pt

# 2. Run inference on images or video
python main.py infer --model models/checkpoints/yolo26n_seg/weights/best.pt --image path/to/image.jpg

# 3. Launch interactive web dashboard
python main.py app
```

---

## Condition classifier

Checkpoint: `models/checkpoints/condition_resnet18.pt` (ResNet18 trained on crops
sampled from the Coralscapes masks — live/bleached/dead coral, up to 250 crops per
class).

Retraining is available via the local entrypoint:
```bash
python main.py train-condition [--manifest ...] [--raw-classes ...] [--crops ...] [--checkpoint ...] [--per-class N] [--epochs N] [--batch N]
```
which calls `src/models/train_condition.py:make_crops()` (crop extraction from
segmentation masks) followed by `train_condition()` (ResNet18 training). The
code lives in `src/models/train_condition.py`.

## Evaluation

Entry point: `python main.py evaluate` → `src/evaluate/segmentation.py:evaluate()`

For each row in `manifest.csv` (filtered to `split == "test"`, optionally truncated by `--max-images N`):

1. Load the image with cv2 and the raw mask via `broad_mask()` to get broad ground-truth class IDs.
2. Predict with Ultralytics at `imgsz=512` matching the model training resolution.
3. Produce a confusion matrix: unsegmented valid pixels are mapped to `other` to account for recall.
4. Compute per-class IoU, mean IoU (mIoU), and Expected Calibration Error (ECE) across detection confidences.

Emitted report: `reports/segmentation_metrics.json` (contains `images`, `confidence_threshold`, `mIoU`, `per_class_iou`, `calibration` with binned reliability stats, and `confusion_matrix`).

## Inference & reef health pipeline

`src/inference/pipeline.py` (`CoralPipeline`):

- `__init__(model_path, condition_model_path=None, imgsz=512, conf=0.20, thresholds=None)`:
  - Loads YOLO26 instance segmentation checkpoint.
  - Optionally loads ResNet-18 condition classifier (`models/checkpoints/condition_resnet18.pt`) for crop-level condition verification.
- `infer(image_path, save_overlay=None)`:
  - `assess_image()` computes blur, brightness, contrast, and color-cast quality signals.
  - `model.predict` generates segmentation masks, bounding boxes, classes, and confidence scores.
  - Computes **Quantitative Ecological Health Indicators**:
    - **Live Coral Cover (LCC %)**: Primary ecological benthic health metric.
    - **Bleaching Prevalence Ratio (%)**: $\frac{\text{bleached}}{\text{total coral}} \times 100\%$.
    - **Coral Mortality Ratio (%)**: $\frac{\text{dead}}{\text{total coral}} \times 100\%$.
    - **Macroalgal Competition**: Coral-to-Algae ratio and algal cover.
    - **Reef Health Category**: Gomez / Reef Check standard tiers (*Excellent* $\ge 50\%$, *Good* $25-50\%$, *Fair* $10-25\%$, *Poor/Degraded* $< 10\%$).
  - Evaluates coral crops through the ResNet-18 condition classifier (when loaded) for crop-level verification.
  - `prediction_uncertainty()` evaluates binary entropy across detection confidences.
  - `review_decision()` transparently determines `accepted` vs `needs_review` status.
  - Annotates overlay image and saves comprehensive JSON report.
- `infer_video()`: extracts frames, computes health metrics per frame, and produces an aggregated video survey report.

## Trust / reliability gate

`src/trust/reliability.py`:

- `expected_calibration_error(confidence, correct, bins=10)`: Computes ECE and binned calibration metrics.
- `softmax(logits, temperature)`: Temperature-scaled probabilities for post-hoc calibration.
- `prediction_uncertainty(scores)`: Normalized entropy over detection confidence scores (avoids penalizing diverse reef compositions).
- `review_decision(quality, confidence, coverage, uncertainty, thresholds)`:
  - Transparently checks image quality flags (`blur`, `exposure`, `low_contrast`, `color_cast`).
  - Evaluates prediction reliability (`low_confidence`, `low_predicted_coverage`, `high_uncertainty`).
  - Emits explicit status (`accepted` or `needs_review`) with detailed flag audit trail.

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
│   └── training_run.json
├── outputs/
│   ├── final_sample/overlay.jpg, report.json
│   └── final_video_inference/
└── project/ , papers/
```

## Reproducibility record

`reports/training_run.json` documents the training parameters and environment:

```json
{
  "project": "Trustworthy AI-Assisted Coral Reef Health Assessment",
  "dataset": {"name":"Coralscapes","prepared_manifest":"...","images":2075,"by_split":{...},"classes":"configs/labels.yaml"},
  "segmentation": {"architecture":"Ultralytics YOLO26n-seg","checkpoint":"models/checkpoints/yolo26n_seg/weights/best.pt","trainer":"Kaggle T4","epochs":30,"imgsz":512,"batch":16,"operating_confidence":0.20,"metrics_mAP50_mask":0.1652},
  "evaluation_reports":["reports/segmentation_metrics.json"]
}
```

## Operational summary

- The Hugging Face Parquet snapshot (`datasets/coralscapes_hf/`) is the canonical dataset source.
- The Kaggle T4 checkpoint at `models/checkpoints/yolo26n_seg/weights/best.pt` is the active segmentation model.
- Segmentation model training runs on Kaggle GPU, while data preparation, evaluation, inference, and the interactive dashboard execute locally.
