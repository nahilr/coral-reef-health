# Trustworthy AI-Assisted Coral Reef Health Assessment

This project uses computer vision to estimate visible benthic cover and coral condition in underwater images and sampled ROV video. It checks image quality and model confidence, then reports a reef-health estimate and a review status.

The project uses one **7-class taxonomy** and six model backends: YOLO26n-sem, YOLO26s-sem, YOLO26m-sem, YOLO26n-seg, SegFormer-B5, and DINOv3-DPT. Run them with the `main.py` command-line tool or the Streamlit app. The pipeline can also process sampled video frames.

---

## Table of contents

1. [Quick start](#quick-start)
2. [Run the project](#run-the-project)
   - [Environment setup](#step-1-environment-setup)
   - [Model weights and checkpoints](#step-2-model-weights--checkpoints)
   - [Image and video inference](#step-3-run-inference-image--video)
   - [Streamlit dashboard](#step-4-launch-the-streamlit-interactive-dashboard)
   - [Prepare the dataset](#step-5-prepare-dataset-from-scratch-optional)
   - [Evaluate models](#step-6-reproduce-benchmark-evaluation)
   - [Train on Kaggle](#step-7-cloud-training-on-kaggle-optional)
   - [CLI reference](#cli-reference-mainpy)
3. [System Architecture and Methods](#system-architecture-and-methods)
   - [Architecture and system flow](#architecture--system-flow)
   - [Dataset and taxonomy](#dataset--7-class-taxonomy)
   - [Health assessment](#inference--quantitative-reef-health-assessment)
   - [Overlays and video](#visual-overlay--video-processing)
   - [Trust and reliability](#trust--reliability-gate)
4. [Evaluate model performance](#evaluate-model-performance)
   - [Detailed benchmark results](model_comparison.md)
   - [Evaluation protocol](#evaluation-protocol--benchmarking-methodology)
5. [Repository layout](#repository-layout)
6. [Glossary](#glossary--terminology)

## Quick start

Use this path to run one image with the default YOLO26n-sem model. Install Git LFS before you clone the repository so Git can download the model checkpoint. These commands use a Bash shell. On Windows, use WSL2 or follow the full setup guide.

```bash
git lfs install
git clone https://github.com/nahilr/coral-reef-health.git
cd coral-reef-health
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py infer --image path/to/reef.jpg --out outputs/quickstart
```

The result folder contains an overlay image and a JSON report. See [Environment setup](#step-1-environment-setup) and [Model weights and checkpoints](#step-2-model-weights--checkpoints) for other backends, data preparation, and optional pretrained weights.

---

## Run the project

Follow this complete step-by-step guide to clone, set up, and run the pipeline on any machine.

### Step 1: Environment Setup

System Requirements:
- Linux / macOS / Windows (WSL2 recommended on Windows).
- Python 3.10 to 3.14.
- NVIDIA GPU with CUDA support recommended (CPU execution supported).

```bash
# 1. Clone the repository
git clone https://github.com/nahilr/coral-reef-health.git
cd coral-reef-health

# 2. Create and activate a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install core dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 2: Model Weights & Checkpoints

The fine-tuned YOLO checkpoints are tracked with Git LFS:
- `models/checkpoints/yolo26n_sem/weights/best.pt` (3.4 MB)
- `models/checkpoints/yolo26s_sem/weights/best.pt` (13.2 MB)
- `models/checkpoints/yolo26m_sem/weights/best.pt` (28.9 MB)
- `models/checkpoints/yolo26n_seg/weights/best.pt` (6.6 MB)

To add another compatible YOLO checkpoint to the app, place its `.pt` file anywhere under `models/checkpoints/` (for example, `models/checkpoints/yolo26m_sem/weights/best.pt`). The app discovers `.pt` files there and reads the model name, task, class names, and saved training image size from the checkpoint. The checkpoint must use the project's 7-class taxonomy for reef-health outputs to be meaningful. Refresh or restart Streamlit after adding the file.

#### Add a model trained on Kaggle

After a Kaggle training run finishes:

1. Download the run artifacts into a new folder under `outputs/kaggle/`:
   ```bash
   kaggle kernels output username/kernel-slug -p outputs/kaggle/my_model_run/
   ```
   Replace `username/kernel-slug` and `my_model_run` with the Kaggle kernel slug and a descriptive local folder name.
2. Find the trained checkpoint and inspect the run results:
   ```bash
   find outputs/kaggle/my_model_run -path '*/weights/best.pt' -print
   ```
   Keep `results.csv`, the run log, and any metrics JSON with the downloaded artifacts. Use the **official test** metrics for comparisons; label validation-only scores as validation results.
3. Copy the best checkpoint into the app's model folder. Replace the source path below with the one printed by `find`:
   ```bash
   mkdir -p models/checkpoints/my_model/weights
   cp outputs/kaggle/my_model_run/checkpoints/run_name/weights/best.pt \
      models/checkpoints/my_model/weights/best.pt
   ```
4. Restart or refresh the Streamlit app. The checkpoint will appear in its model selector, with its architecture, task, class names, and saved image size read from the weights file. Only add a checkpoint trained with the project's 7-class taxonomy; a different or reordered class map can make the reef-health results incorrect.
5. Run the model on the same official test split and use the same evaluation protocol as the other models. Record its training details and results in `model_comparison.md`. Do not compare a validation score with a test score. The `models/README.md` file lists checkpoints and links to the report.

If git cloned pointer files instead of binaries, pull them via:
```bash
git lfs pull
```

*(Optional)* To download the public SegFormer-B5 and DINOv3-DPT pretrained backbones (~1.6 GB):
```bash
python scripts/fetch_pretrained.py
```
*Note: Network access required; downloads weights from EPFL-ECEO Hugging Face repos and configures offline ViT loaders.*

---

### Step 3: Run Inference (Image & Video)

#### A. Single Image Inference
Run inference on any underwater reef photo:
```bash
# Real-time edge semantic model (default: YOLO26n-sem)
python main.py infer --image sample.jpg --out outputs/inference

# High-precision YOLO26s-sem model
python main.py infer --model models/checkpoints/yolo26s_sem/weights/best.pt \
                     --image sample.jpg --out outputs/inference

# DINOv3 foundation model
python main.py infer --model models/pretrained/coralscapesv2-dinov3-vitl-lora-dpt-95_class \
                     --model-type dinov3 --image sample.jpg --out outputs/inference
```
**Outputs generated in `outputs/inference/`**:
- `overlay.jpg`: The original image with color-coded benthic masks, detection boxes, and the bottom overlay banner.
- `report.json`: Machine-readable JSON containing quality flags, mean confidence, benthic area percentages, coral condition, cover tier, and a combined category.

#### B. Video Survey Inference
Process underwater ROV survey footage with frame subsampling and survey-level aggregation:
```bash
# Process every 2nd frame of an underwater transect video
python main.py infer --video underwater_transect.mp4 --every-n 2 --out outputs/video_run
```
**Outputs generated in `outputs/video_run/`**:
- `annotated_survey.mp4`: Compressed H.264 video with real-time bottom banner telemetry rendered on every processed frame.
- `report.json`: Survey-wide aggregated metrics (mean LCC, mean bleaching ratio, acceptance rate, and coral condition), with per-frame details. It also retains the combined `overall_health_category` field.

#### C. Batch inference from a folder
Use `--dir` to recursively process supported images and videos in one folder. The model is loaded once for the batch. Annotated media, per-file JSON reports, and the batch manifest are written into the folder passed to `--out`, preserving any nested input subfolders.

#### Curated test-set examples
Selected examples spanning cover, bleaching, mortality, algae, sand-only, and review cases are collected in `data/processed/showcase_candidates/`. Its `README.md` explains the selection and cached-report status; `index.csv` summarizes metrics and reliability outcomes.

```bash
python main.py infer \
  --dir path/to/small-test-folder \
  --out outputs/batch_run \
  --model models/checkpoints/yolo26s_sem/weights/best.pt \
  --model-type yolo \
  --imgsz 1024 \
  --conf 0.20 \
  --every-n 1
```

The folder scan supports `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.mp4`, `.avi`, `.mov`, and `.mkv` files. A flat input folder produces annotated image overlays, videos, per-file JSON reports, and `manifest.json` directly in `outputs/batch_run/`. `--every-n` controls video frame sampling; `1` processes every frame. To smoke-test the batch workflow, pass a small folder containing only a few images or one short video.

---

### Step 4: Launch the Streamlit Interactive Dashboard

Launch the browser interface for visual pair-analysis:
```bash
python main.py app --port 8501
```
Open `http://localhost:8501` in your browser to:
- Switch backends on the fly (YOLO26n-sem, YOLO26s-sem, YOLO26m-sem, YOLO26n-seg, SegFormer-B5, DINOv3-DPT).
- Upload images or underwater videos with live on-the-fly inference playback, real-time telemetry, and downloadable annotated video streams.
- Inspect the interactive benthic composition breakdown table and quality gate alerts.
- Download structured JSON health reports and annotated video streams.

---

### Step 5: Prepare Dataset from Scratch (Optional)

If you have downloaded the raw Coralscapes Parquet dataset from Hugging Face (`josauder/coralscapes`):
```bash
python main.py prepare --root data/raw/coralscapes --out data/processed/yolo
```
This will:
1. Decode raw RGB images and 95-class segmentation masks.
2. Project 95 fine-grained labels to the 7-class taxonomy using `configs/label_map.json`.
3. Extract connected components and emit normalized YOLO polygon label files.
4. Generate `data.yaml` and `manifest.csv` tracking split distributions.

---

### Step 6: Reproduce Benchmark Evaluation

Evaluate any checkpoint on the held-out test split:
```bash
# Evaluate YOLO26n-sem
python main.py evaluate --model models/checkpoints/yolo26n_sem/weights/best.pt \
                        --manifest data/processed/yolo/manifest.csv \
                        --out reports/eval_metrics.json

# Run the evaluator on the official 473-image test split (all six models)
python scripts/evaluate_model_comparison.py
```

---

### Step 7: Cloud Training on Kaggle (Optional)

The repository provides standalone Kaggle training kernels under `kaggle/` that stream data directly from Hugging Face without requiring 7 GB of local storage:

```bash
# Push and trigger training kernels via Kaggle CLI
kaggle kernels push -p kaggle/yolo26n_sem/
kaggle kernels push -p kaggle/yolo26s_sem/
kaggle kernels push -p kaggle/yolo26n_seg/
kaggle kernels push -p kaggle/yolo26m_sem/
```
The kernels train for 30–85 epochs on an NVIDIA T4 GPU and output `best.pt` along with validation curves and metrics.

---

### Step 8: Run Automated Test Suite

Verify system integrity with 29 comprehensive unit tests:
```bash
# With pytest
pytest

# Without any external test dependencies (pure standard library runner)
python tests/run_all.py
```

---

### CLI Reference: `main.py`

`main.py` provides a unified interface with 4 explicit subcommands:

### 1. Inference (`infer`)
```bash
python main.py infer [-h] [--model MODEL] [--model-type {yolo,segformer,dinov3}]
                     [--image IMAGE] [--video VIDEO] [--dir INPUT_DIR] [--out OUT]
                     [--imgsz IMGSZ] [--conf CONF] [--every-n EVERY_N]
                     [--max-frames MAX_FRAMES] [--save-frames]
                     [--output-video-name OUTPUT_VIDEO_NAME]
```

### 2. Evaluation (`evaluate`)
```bash
python main.py evaluate [-h] [--model MODEL] [--manifest MANIFEST]
                        [--raw-classes RAW_CLASSES] [--max-images MAX_IMAGES]
                        [--conf CONF] [--imgsz IMGSZ] [--out OUT]
```

### 3. Data Preparation (`prepare`)
```bash
python main.py prepare [-h] [--root ROOT] [--out OUT]
                       [--max-images-per-split MAX_IMAGES_PER_SPLIT]
```

### 4. Interactive UI (`app`)
```bash
python main.py app [-h] [--port PORT]
```

---

## System Architecture and Methods

### Architecture & System Flow

```
                  ┌────────────────────────────────────────┐
                  │       Underwater Image / Video         │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │       Image Quality Gate               │
                  │   Blur, Exposure, Contrast, Color Cast │
                  └───────────────────┬────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
    ┌───────────────────────────────┐   ┌───────────────────────────────┐
    │     Dense Semantic Models     │   │      Instance Models          │
    │  • YOLO26n-sem                │   │  • YOLO26n-seg                │
    │  • YOLO26s-sem                │   └───────────────┬───────────────┘
    │  • YOLO26m-sem                │                   │
    │  • SegFormer-B5               │                   │
    │  • DINOv3 ViT-L + DPT         │                   │
    └───────────────┬───────────────┘                   │
                    │                                   │
                    ▼                                   ▼
    ┌───────────────────────────────┐   ┌───────────────────────────────┐
    │  Dense 7-Class Prediction     │   │  Instance Boxes & Masks       │
    │  + Shannon Entropy            │   │  + Detection Confidence       │
    └───────────────┬───────────────┘   └───────────────┬───────────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │     Connected-Component Region Boxes   │
                  │    Extracts Display Boxes for Coral    │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │     Health Indicator Engine            │
                  │  • Live Coral Cover (LCC %)            │
                  │  • Bleaching & Mortality Ratios (%)    │
                  │  • Algae Competition Index             │
                  │  • Condition-Capped Health Tiers       │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │     Reliability & Review Gate          │
                  │   Quality + Uncertainty + Coverage     │
                  │     ACCEPTED vs NEEDS EXPERT REVIEW    │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │               Outputs                  │
                  │ Overlay Image • JSON Report • MP4 Video│
                  └────────────────────────────────────────┘
```

---

### Dataset & 7-Class Taxonomy

### Taxonomy Definitions & Ecological Roles

The project collapses the fine-grained, long-tailed Coralscapes classes into **7 unified ecological classes** (`configs/labels.yaml`):

| ID | Class Name | Substrate Category & Ecological Role | BGR Overlay Color | RGB Hex |
|:---:|---|---|---|:---:|
| **0** | `coral_alive` | Living coral colonies of all morphologies (acropora, massive, brain, etc.) | `[70, 170, 70]` | `#46aa46` |
| **1** | `coral_bleached` | Stressed, living colonies exhibiting visible thermal pigment loss | `[255, 230, 80]` | `#50e6ff` |
| **2** | `coral_dead` | Dead coral structure, intact skeleton, and recently dead colonies | `[150, 80, 80]` | `#505096` |
| **3** | `algae` | Macroalgae, turf algae, and algae-covered dead substrate | `[0, 160, 120]` | `#78a000` |
| **4** | `hard_substrate` | Consolidated limestone pavement, dead rock, boulders, and rubble | `[140, 130, 120]` | `#78828c` |
| **5** | `sand` | Unconsolidated soft sediments, sand patches, and silt | `[194, 178, 128]` | `#80b2c2` |
| **6** | `non_reef` | Water column, open blue water, fish, transect tape, and hardware | `[30, 30, 30]` | `#1e1e1e` |

* **Reef Denominator**: Classes 0–5 define the valid benthic seafloor.
* **Excluded Sink**: Class 6 (`non_reef`) is strictly excluded from health ratios so the amount of blue water in a photograph does not inflate or deflate benthic cover percentages.

### Fine-Grained Label Mapping Logic

The mapping is explicitly defined in `configs/label_map.json`:
- **Coral Condition Preservation**: `acropora bleached` → `coral_bleached`, `massive dead` → `coral_dead`, `brain coral alive` → `coral_alive`.
- **Substrate Unification**: `rubble`, `hard substrate`, `dead giant clam` → `hard_substrate`.
- **Algae Aggregation**: `algae covered substrate`, `algae covered table acropora` → `algae`.
- **Non-Benthic Sinks**: `fish`, `sea cucumber`, `water column`, `transect line`, `trash` → `non_reef`.

### Component-Derived Instance Polygons

In `src/data/prepare.py`, binary class masks are processed via `cv2.findContours(..., cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`. Polygons are filtered to a minimum area threshold and normalized to [0, 1] coordinates for training instance segmentation backends (`yolo26n_seg`).

---

### Inference & Quantitative Reef Health Assessment

### Core Ecological Metric Formulations

Quantitative indicators are computed by `compute_health_indicators()` in `src/inference/pipeline.py`:

$$\text{Reef Substrate Area} = \text{Total Valid Pixels} - \text{Pixels}(\text{non-reef})$$

1. **Live Coral Cover (LCC %)**:
   $$\text{LCC} = \frac{\text{Pixels}(\text{coral-alive})}{\text{Reef Substrate Area}} \times 100$$
2. **Bleaching Ratio (%)**:
   $$\text{Bleaching Ratio} = \frac{\text{Pixels}(\text{coral-bleached})}{\text{Pixels}(\text{coral-alive}) + \text{Pixels}(\text{coral-bleached}) + \text{Pixels}(\text{coral-dead})} \times 100$$
3. **Mortality Ratio (%)**:
   $$\text{Mortality Ratio} = \frac{\text{Pixels}(\text{coral-dead})}{\text{Pixels}(\text{coral-alive}) + \text{Pixels}(\text{coral-bleached}) + \text{Pixels}(\text{coral-dead})} \times 100$$
4. **Coral-to-Algae Competition Ratio**:
   $$\text{Coral-to-Algae Ratio} = \frac{\text{Pixels}(\text{coral-alive})}{\max(1, \text{Pixels}(\text{algae}))}$$

---

### Coral Condition and Live Cover

The image, video, and survey displays show **Coral Condition** separately from Live Coral Cover (LCC). Coral condition is based on the bleaching and dead-coral ratios among detected coral. If no coral is detected, it is reported as **Not assessable (No coral detected)**. A sand-only image can therefore have 0% LCC and a **Poor** cover tier while coral condition remains not assessable.

The cover tiers are **Excellent**, **Good**, **Fair**, and **Poor**; **Poor** means LCC below 10%. The app displays the condition and LCC separately. The JSON `reef_health_category` field remains a combined summary for compatibility; it applies the cover tier, bleaching/mortality caps, and an algae warning. Use `coral_condition` for coral condition and `live_cover_tier` for the cover band.

### Combined Reef Assessment in JSON

The verdict combines two independent ecological axes:
$$\text{Verdict} = \mathbf{Category} \quad [\mathbf{Condition}]$$

#### Category Divisioning & LCC Thresholds
The base category follows the international Reef Check / Gomez et al. standard:
- **`Excellent`**: Live Coral Cover ≥ 50.0%
- **`Good`**: Live Coral Cover 25.0% – 49.9%
- **`Fair`**: Live Coral Cover 10.0% – 24.9%
- **`Poor`**: Live Coral Cover < 10.0%
- **`Indeterminate`**: No reef substrate detected in frame (0% benthic area).

#### Condition Disturbance Divisioning & Capping Guardrail
When thermal stress or mortality is detected, disturbance flags are raised and **cap** the reported health tier downwards. **Disturbance can only lower a tier, never raise it.**

| Disturbance Axis | Ratio Range | Condition Label | Maximum Allowed Tier |
|---|:---:|---|:---:|
| **Bleaching** | 10% – 25% | `Moderate Bleaching Warning` | **Good** |
| | 25% – 50% | `Severe Bleaching Alert` | **Fair** |
| | ≥ 50% | `Extensive Bleaching` | **Poor** |
| **Mortality** | 10% – 25% | `Elevated Mortality` | **Good** |
| | 25% – 50% | `High Mortality` | **Fair** |
| | ≥ 50% | `Extensive Mortality` | **Poor** |
| **Algae Overgrowth** | Algae > 30% and Algae > LCC | `Algae-Covered Area Warning` | Preserves tier, flags phase shift |

#### Examples
- `reef_health_category: Excellent`: LCC ≥ 50%, with low bleaching and dead-coral ratios.
- `reef_health_category: Good`: LCC = 38%, with low bleaching and dead-coral ratios.
- `reef_health_category: Fair [Severe Bleaching Alert]`: the cover tier is capped because the bleaching ratio reaches the severe threshold.
- `reef_health_category: Poor [Extensive Mortality]`: the cover tier is capped because the dead-coral ratio reaches the extensive threshold.
- A sand-only image can have `live_cover_tier: Poor` and `coral_condition: Not assessable (No coral detected)`.

---

### Visual Overlay & Video Processing

### Bottom Overlay Banner

The default overlay renders a banner along the bottom ~8–10% of the frame, leaving the reef scene clear:

```text
+================================================================================================================+ <- 2px Accent Border
| CORAL CONDITION: Stable  |  LCC: 38.5% · Bleached: 2.1% · Algae: 12.0%   [ Model: YOLO26n-sem | ACCEPTED ]  | <- Row 1: Telemetry
| [🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟨🟫🟫🟩🟩🟩🟩⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜🟨🟨🟨🟨🟨🟨🟨🟨]                        | <- Row 2: Stacked Bar
| ■ Alive 32.7%   ■ Bleached 1.8%   ■ Dead 3.6%   ■ Algae 10.2%   ■ Substrate 20.4%   ■ Sand 16.3%   ■ Non-Reef 15.0% | <- Row 3: Color Guide
+----------------------------------------------------------------------------------------------------------------+
```

1. **Top Accent Line**: 2px border color-coded to coral condition (stable, warning, alert, or not assessable).
2. **Row 1 (Telemetry)**: Displays Coral Condition, LCC, bleached cover, algae cover, active Model name, Trust status, Coverage %, and Confidence.
3. **Row 2 (Stacked Composition Bar)**: Horizontal progress bar proportioned to exact pixel area fractions in the 7 class colors.
4. **Row 3 (Color Guide Legend)**: Swatches show each class's share of classified image pixels. Row 1 health values use their ecological denominators.

### Overlay Blending Formula & Color Guide
Masks are blended onto the original image using:
$$\text{Overlay}[x, y] = 0.55 \times \text{Image}[x, y] + 0.45 \times \text{ClassColor}$$

### Bounding Box Region Extraction & Label Collision Avoidance
For dense models, display bounding boxes are synthesized around connected coral components (`BOX_CLASSES = ('coral_alive', 'coral_bleached', 'coral_dead')`). A collision avoidance algorithm tests slots (above, inside-top, inside-bottom) and reserves the bottom banner region to prevent text overlap.

### Video Streaming & Survey-Level Aggregation
In `infer_video`, frames are sampled via `--every-n`, overlay HUDs are rendered, and streams are encoded using `ffmpeg` (`-c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p`). The app shows the aggregated coral condition and cover metrics. The JSON summary also retains `overall_health_category`, the combined cover-and-condition field.

---

### Trust & Reliability Gate

Every inference pass executes automated verification (`src/trust/reliability.py`):

### Image Quality Assessment Thresholds
Implemented in `src/data/quality.py`:
- **Laplacian Blur Variance**: flags blur when variance < 12.0.
- **Luminance Exposure**: flags under/over-exposure when mean luminance ∉ [25, 235].
- **Contrast Deviation**: flags low contrast when standard deviation < 18.0.
- **Underwater Color Cast**: flags color cast when max/min channel mean ratio > 2.2.

### Uncertainty & Coverage Floors
- **Benthic Coverage**: flags `low_predicted_coverage` if assessable seafloor < 5.0%.
- **Shannon Entropy / Confidence**: flags `low_confidence` or `high_uncertainty` when model probabilities fall below calibration thresholds.

### Auditable Decision States
- **`ACCEPTED`**: Image quality and model certainty meet all criteria for automated reporting.
- **`NEEDS EXPERT REVIEW`**: Triggers explicit review flags (e.g. `['blur', 'low_predicted_coverage']`), routing the sample to expert human review.

---

## Evaluate model performance

### Quick model comparison

All scores use the same 473-image Coralscapes test split and the project 7-class taxonomy.

| Model | Task | Macro mIoU | Pixel accuracy |
|---|---|---:|---:|
| DINOv3 ViT-L + LoRA + DPT | Semantic | **72.65%** | **87.20%** |
| SegFormer-B5 | Semantic | 67.87% | 85.36% |
| YOLO26m-sem | Semantic | 60.23% | 81.13% |
| YOLO26s-sem | Semantic | 57.98% | 79.97% |
| YOLO26n-sem | Semantic | 51.94% | 75.64% |
| YOLO26n-seg | Instance, rasterized for scoring | 41.04% | 52.32% |

Use semantic models to measure benthic cover and health ratios. Use instance segmentation to inspect individual masks. This project's instance labels come from connected components in semantic masks, so they do not support validated colony counts.

See [model_comparison.md](model_comparison.md) for per-class scores, health-output agreement, training runs, speed measurements, and the full evaluation protocol.

### Evaluation Protocol & Benchmarking Methodology

Evaluations in `scripts/evaluate_model_comparison.py` adhere to strict reproducibility guidelines:
1. **Official Split Only**: Evaluated on all 473 official Coralscapes test images at 2048 × 1024 source resolution.
2. **Logit Folding**: Dense models outputting 95 raw classes have fine probabilities summed into the 7 project classes before taking the argmax.
3. **Void Pixel Handling**: Any pixel predicted as void/unmapped counts as an incorrect (missed) prediction.
4. **Macro mIoU Calculation**: Macro IoU averages classes with a non-zero union over valid ground-truth pixels.
5. **Warning on `--max-images N`**: This parameter takes the first $N$ manifest rows as a pipeline smoke test and must not be cited as a model performance metric.

---

## Repository Layout

```
.
├── model_comparison.md                # Detailed benchmark results
├── configs/
│   ├── labels.yaml                     # Canonical 7-class project taxonomy & roles
│   ├── label_map.json                  # Explicit 95 -> 7 raw class mapping
│   └── dinov3_id2label.json            # 95-class ID-to-name mapping for DINOv3
├── kaggle/
│   ├── yolo26n_sem/                    # Kaggle kernel for YOLO26n-sem (30 epochs)
│   ├── yolo26s_sem/                    # Kaggle kernel for YOLO26s-sem (50 epochs)
│   ├── yolo26m_sem/                    # Kaggle kernel for YOLO26m-sem (85 epochs)
│   ├── yolo26n_seg/                    # Kaggle kernel for YOLO26n-seg (instances)
│   └── README.md                       # Kaggle execution notes
├── models/
│   ├── checkpoints/
│   │   ├── yolo26n_sem/weights/best.pt # Fine-tuned 7-class YOLO26n-sem (Git LFS)
│   │   ├── yolo26s_sem/weights/best.pt # Fine-tuned 7-class YOLO26s-sem (Git LFS)
│   │   ├── yolo26m_sem/weights/best.pt # Fine-tuned 7-class YOLO26m-sem (Git LFS)
│   │   └── yolo26n_seg/weights/best.pt # Fine-tuned 7-class YOLO26n-seg (Git LFS)
│   ├── pretrained/                     # SegFormer & DINOv3 backbones (via fetch_pretrained.py)
│   └── model_comparison.json           # Machine-readable benchmark metrics
├── scripts/
│   ├── evaluate_model_comparison.py    # Reproducible 6-backend evaluator
│   └── fetch_pretrained.py             # Downloader for public SegFormer & DINOv3 weights
├── src/
│   ├── app/
│   │   └── streamlit_app.py            # Streamlit interactive dashboard
│   ├── data/
│   │   ├── label_map.py                # Taxonomy mappings and mask projection
│   │   ├── prepare.py                  # Parquet dataset preparation
│   │   └── quality.py                  # Blur, exposure, contrast, color cast checks
│   ├── evaluate/
│   │   └── segmentation.py             # Evaluation routines and mIoU metrics
│   ├── inference/
│   │   ├── dinov3_dpt.py               # DINOv3 loader with offline ViT reconstruction
│   │   └── pipeline.py                 # Multi-backend CoralPipeline & bottom banner HUD
│   └── trust/
│       └── reliability.py              # Review decision gate, ECE, entropy
├── tests/
│   ├── test_dense_detections.py        # Region bounding box and gate tests
│   ├── test_health_indicators.py       # Reef health verdict and condition cap tests
│   ├── test_label_map.py               # Taxonomy mapping integrity tests
│   ├── test_quality.py                 # Image quality gate and BGR consistency tests
│   ├── test_video_pipeline.py          # Video-to-video inference test
│   └── run_all.py                      # Standalone test runner
├── main.py                             # Unified CLI entry point (infer, evaluate, prepare, app)
└── requirements.txt                    # Minimal python dependencies
```

---

## Comprehensive Glossary & Terminology

### 1. Ecological & Benthic Health Metrics

| Term / Abbreviation | Full Name | Definition & Ecological Relevance |
|---|---|---|
| **LCC (%)** | Live Coral Cover Percentage | The primary metric in coral reef ecology: $`\frac{\text{Live Coral Pixels}}{\text{Reef Substrate Area}} \times 100\%`$. Measures living, non-bleached reef-building coral. |
| **Bleaching Ratio (%)** | Coral Bleaching Proportion | Ratio of bleached coral to total coral colony area: $`\frac{\text{Bleached}}{\text{Alive} + \text{Bleached} + \text{Dead}} \times 100\%`$. Primary signal for marine heatwaves. |
| **Mortality Ratio (%)** | Coral Mortality Proportion | Ratio of dead coral structure to total coral colony area: $`\frac{\text{Dead}}{\text{Alive} + \text{Bleached} + \text{Dead}} \times 100\%`$. Indicates recent or chronic mortality. |
| **Coral-to-Algae Ratio** | Competition Index | Ratio of live coral pixels to algal pixels. A ratio < 1.0 indicates algal dominance and potential phase shift. |
| **Benthos / Benthic** | Seafloor Substrate | The ecological zone at the seabed, comprising living coral, hard substrate, sand, and flora attached to the ocean bottom. |
| **Reef Substrate Denominator** | Assessable Seafloor Area | Total valid image pixels minus open water column, fish, and hardware. Prevents water from deflating coral percentages. |

---

### 2. Computer Vision & Calibration Metrics

| Metric | Full Name | Definition |
|---|---|---|
| **Macro mIoU** | Mean Intersection over Union | Area of overlap between prediction and ground truth divided by area of union, averaged equally across all valid classes. |
| **Pixel Accuracy** | Global Pixel Accuracy | Percentage of valid pixels correctly classified: $`\frac{\sum \text{True Positives}}{\text{Total Valid Pixels}}`$. |
| **ECE** | Expected Calibration Error | Difference between model confidence and actual accuracy across binned confidence intervals; measures reliability. |
| **Mask mAP@50** | Mask Mean Average Precision | Instance segmentation precision evaluated at an IoU threshold of 0.50 over predicted polygon masks. |

---

### 3. Model Architectures & Paradigms

| Architecture / Term | Paradigm | Description |
|---|---|---|
| **Semantic Segmentation (`sem`)** | Dense Pixel Classification | Classifies every pixel (x, y) in the image into one of 7 classes; ideal for calculating benthic surface areas. |
| **Instance Segmentation (`seg`)** | Object Detection + Polygons | Predicts discrete bounding boxes and polygon masks per colony; ideal for colony counting and size profiling. |
| **DINOv3 ViT-L + DPT** | Vision Foundation Model | Large vision transformer with Low-Rank Adaptation (LoRA) and Dense Prediction Transformer (DPT) head. |
| **SegFormer-B5** | Hierarchical Transformer | Transformer architecture designed for dense semantic segmentation at high resolution. |
| **YOLO26n/s/m-sem** | Real-Time Edge CNN | Ultralytics models for pixel classification on live ROV video, with different size/speed trade-offs. See the [benchmark report](model_comparison.md) for measured speed. |

---

### 4. Trust & Reliability Engineering

| Term | Concept | Operational Role |
|---|---|---|
| **Quality Gate** | Pre-Inference Screening | Algorithms checking blur variance, luminance exposure, contrast, and color cast before model execution. |
| **Normalized Shannon Entropy** | Prediction Uncertainty | Multi-class prediction entropy normalized to [0, 1]; flags ambiguous or unfamiliar underwater scenes. |
| **Abstention / Review Flag** | Safe AI Fallback | Automatically routing uncertain or low-quality imagery to expert human review rather than outputting false data. |
| **Condition Capping** | Ecological Safety Guardrail | Rule-based constraint preventing a heavily bleached or dead reef assemblage from being graded as "Healthy" or "Excellent". |
