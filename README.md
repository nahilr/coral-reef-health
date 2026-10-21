# Trustworthy AI-Assisted Coral Reef Health Assessment

A reproducible computer vision pipeline for automated benthic composition and visible coral-condition assessment from underwater imagery and sampled video frames. Results include image-quality checks, predicted benthic coverage, backend-specific confidence or uncertainty signals where available, and an explicit `accepted` or `needs_review` decision.

The pipeline operates on a standardized **7-class benthic taxonomy** served by **four interchangeable backends** (YOLO26n-sem, YOLO26n-seg, SegFormer-B5, DINOv3-DPT) through a unified inference interface, an interactive Streamlit UI, and automated video survey aggregation.

---

## Table of contents

1. [Backends & Measured Performance](#backends--measured-performance)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Dataset & 7-Class Taxonomy](#dataset--7-class-taxonomy)
5. [CLI Entry Point: `main.py`](#cli-entry-point-mainpy)
6. [Inference & Reef Health Indicators](#inference--reef-health-indicators)
7. [Trust & Reliability Gate](#trust--reliability-gate)
8. [Streamlit UI](#streamlit-ui)
9. [Kaggle Cloud Training](#kaggle-cloud-training)
10. [Repository Layout](#repository-layout)
11. [Glossary & Terminology](#glossary--terminology)

---

## Backends & Measured Performance

All four backends are evaluated on the **473 images** of the official Coralscapes test split against the 7-class project taxonomy (`configs/labels.yaml`).

| Model | Checkpoint / Source | Task | Pixel Accuracy | Macro mIoU | Inference Speed | Model Size |
|---|---|---|---:|---:|---:|---:|
| **DINOv3 ViT-L + DPT** | `models/pretrained/coralscapesv2-dinov3-vitl-lora-dpt-95_class/` | semantic | **87.20%** | **72.65%** | 3.58 s/image | 1.33 GB |
| **SegFormer-B5** | `models/pretrained/segformer-b5-coralscapesv2-95/` | semantic | 85.36% | 67.87% | 1.30 s/image | 339 MB |
| **YOLO26n-sem** | `models/checkpoints/yolo26n_sem/weights/best.pt` | semantic | 75.64% | 51.94% | **0.026 s/image¹** | 3.4 MB |
| **YOLO26n-seg** | `models/checkpoints/yolo26n_seg/weights/best.pt` | instance → rasterised | 52.32% | 41.04% | 0.032 s/image | 6.6 MB |

¹ Single-image steady-state inference at `imgsz=1024`, batch 1, FP32 on NVIDIA GPU (~38.2 FPS).

### Health Output Agreement (GT-Mask Derived Labels)

| Backend | Live-Cover MAE | Live-Cover Bias | Health Tier Agreement | Within 1 Tier |
|---|---:|---:|---:|---:|
| **DINOv3 ViT-L + DPT** | **2.99 pts** | +0.03 | **80.5%** | **98.5%** |
| **SegFormer-B5** | 3.87 pts | +1.17 | 77.8% | 97.5% |
| **YOLO26n-sem** | 7.16 pts | +2.35 | 64.5% | 93.2% |
| **YOLO26n-seg** | 23.34 pts | +22.67 | 47.8% | 81.0% |

- **YOLO26n-sem** is the fastest backend, running edge-semantic inference in real time (38+ FPS).
- **DINOv3 ViT-L + DPT** provides the highest accuracy and mIoU. Its ViT encoder configuration is rebuilt offline directly from `model.safetensors`, requiring no gated Hugging Face token.
- **SegFormer-B5** provides a strong balanced transformer baseline running in FP32.
- **YOLO26n-seg** provides instance bounding boxes and masks. Its dense score reflects rasterization of instance masks at conf 0.25 over connected components.

---

## Quick Start

### 1. Installation

```bash
git clone <repo-url>
cd BTP
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Fetch Pretrained Foundation Backbones (Optional)

The fine-tuned YOLO checkpoints are versioned in git with Git LFS (`models/checkpoints/`). To download the public SegFormer and DINOv3 backbones (~1.6 GB):

```bash
python scripts/fetch_pretrained.py
```

### 3. Run Inference

Single underwater image:
```bash
python main.py infer --image sample.jpg --out outputs/demo
```

Underwater survey video:
```bash
python main.py infer --video survey.mp4 --every-n 2 --out outputs/video_run
```

Launch the interactive Streamlit dashboard:
```bash
python main.py app
```

Run test suite:
```bash
pytest
# or without pytest:
python tests/run_all.py
```

---

## Architecture

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
    │  • YOLO26n-sem (38+ FPS)      │   │  • YOLO26n-seg (instances)    │
    │  • SegFormer-B5               │   └───────────────┬───────────────┘
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
                  │  HUD Overlay • JSON Report • MP4 Video │
                  └────────────────────────────────────────┘
```

---

## Dataset & 7-Class Taxonomy

The project collapses the fine-grained Coralscapes categories into **7 unified task classes** (`configs/labels.yaml`):

| ID | Class | Substrate / Ecological Role | Overlay Color |
|:---:|---|---|---|
| **0** | `coral_alive` | Living coral colonies (all morphologies) | Green `[70, 170, 70]` |
| **1** | `coral_bleached` | Stressed, bleached coral colonies | White/Yellow `[255, 230, 80]` |
| **2** | `coral_dead` | Dead coral skeleton / structure | Dark Red `[150, 80, 80]` |
| **3** | `algae` | Macroalgae and algal turf | Olive Green `[0, 160, 120]` |
| **4** | `hard_substrate` | Consolidated rock, pavement, and rubble | Slate Stone `[140, 130, 120]` |
| **5** | `sand` | Soft sand sediment | Sand Beige `[194, 178, 128]` |
| **6** | `non_reef` | Water column, fish, transect hardware | Near-Black `[30, 30, 30]` |

The explicit raw-to-project mapping is defined in `configs/label_map.json`. Classes 0–5 define the reef substrate denominator; class 6 (`non_reef`) is excluded so open water or hardware cannot distort benthic cover percentages.

---

## CLI Entry Point: `main.py`

`main.py` is the single entry point for all workflows:

### 1. Inference (`infer`)
```bash
# Image inference with real-time YOLO26n-sem (default)
python main.py infer --image path/to/reef.jpg --out outputs/inference

# Image inference with DINOv3 ViT-L + DPT
python main.py infer --model models/pretrained/coralscapesv2-dinov3-vitl-lora-dpt-95_class \
                     --model-type dinov3 --image path/to/reef.jpg --out outputs/inference

# Video inference (subsampling every 2nd frame)
python main.py infer --video survey.mp4 --every-n 2 --out outputs/survey_run
```

### 2. Evaluation (`evaluate`)
Evaluates a segmentation checkpoint against the 7-class project taxonomy:
```bash
python main.py evaluate --model models/checkpoints/yolo26n_sem/weights/best.pt \
                        --manifest data/processed/yolo/manifest.csv \
                        --out reports/eval_metrics.json
```

### 3. Data Preparation (`prepare`)
Converts raw Coralscapes Parquet snapshots into 7-class YOLO segmentation format:
```bash
python main.py prepare --root data/raw/coralscapes --out data/processed/yolo
```

### 4. Interactive UI (`app`)
Launches the Streamlit web dashboard:
```bash
python main.py app --port 8501
```

---

## Inference & Reef Health Indicators

### Ecological Metrics (`compute_health_indicators`)
1. **Live Coral Cover (LCC %)**:
   $$\text{LCC} = \frac{\text{Pixels}(\text{coral\_alive})}{\text{Reef Substrate Area}} \times 100$$
2. **Bleaching Ratio (%)**:
   $$\text{Bleaching Ratio} = \frac{\text{Pixels}(\text{coral\_bleached})}{\text{Total Coral Area}} \times 100$$
3. **Mortality Ratio (%)**:
   $$\text{Mortality Ratio} = \frac{\text{Pixels}(\text{dead\_coral})}{\text{Total Coral Area}} \times 100$$
4. **Coral-to-Algae Competition Ratio**:
   $$\text{Coral-to-Algae Ratio} = \frac{\text{Pixels}(\text{coral\_alive})}{\max(1, \text{Pixels}(\text{algae}))}$$

### Health Verdict Rule (Condition-Capped Gomez/Reef Check Benchmark)
Base live-cover tiers:
- **Excellent**: $\ge 50\%$
- **Good**: $25\% - 50\%$
- **Fair**: $10\% - 25\%$
- **Poor / Degraded**: $< 10\%$

**Condition Cap Rule**: Bleaching or mortality disturbance strictly **caps** the tier:
- Bleaching or Mortality $\ge 10\%$: capped at **Good** (`Moderate Warning`)
- Bleaching or Mortality $\ge 25\%$: capped at **Fair** (`Severe Alert`)
- Bleaching or Mortality $\ge 50\%$: capped at **Poor / Degraded** (`Extensive Disturbance`)

Condition disturbances can only *lower* the reported category, never inflate it. The pure cover tier remains accessible as `live_cover_tier`.

---

## Trust & Reliability Gate

Every inference pass executes automated verification (`src/trust/reliability.py`):

1. **Image Quality Assessment (`src/data/quality.py`)**:
   - **Laplacian Blur Variance**: flags blur when variance $< 12.0$.
   - **Luminance Exposure**: flags under/over-exposure when mean gray $\notin [25, 235]$.
   - **Contrast Deviation**: flags low contrast when standard deviation $< 18.0$.
   - **Underwater Color Cast**: flags severe color cast when max/min channel mean ratio $> 2.2$.
2. **Model Uncertainty & Coverage**:
   - Confidence and uncertainty depend on the selected backend. Dense transformer outputs provide per-pixel scores and entropy; YOLO instance outputs provide detection confidence; YOLO semantic outputs do not expose a calibrated confidence or entropy map.
   - **Benthic Coverage**: flags `low_predicted_coverage` if assessable substrate $< 5.0\%$.
   - **Confidence Floors**: flags `low_confidence` when the selected backend provides a mean probability below threshold.
3. **Decision State**:
   - `ACCEPTED`: Image quality and prediction certainty satisfy all criteria.
   - `NEEDS EXPERT REVIEW`: Triggers specific flags explaining why expert verification is required.

---

## Streamlit UI

Launch the web app via `python main.py app`.

Features:
- **Model Selector**: Switch between YOLO26n-sem, YOLO26n-seg, SegFormer-B5, and DINOv3-DPT on the fly.
- **Single Image Analysis**: Side-by-side original and HUD overlay display, trust banner, Reef Check metrics, benthic area breakdown table, and JSON report download.
- **Underwater Video Analysis**: Full or subsampled video processing (`every_n`), live progress bar, survey-level aggregated health indicators, MP4 stream playback, sampled keyframe carousel, and annotated video download.

---

## Kaggle Cloud Training

The repository includes standalone Kaggle training scripts for the 7-class YOLO models under `kaggle/`:

- `kaggle/yolo26n_sem/train_kaggle.py` (`nahilr/coral-btp-yolo-sem`): Trains native 7-class YOLO26n-sem for 30 epochs (mIoU 0.5194).
- `kaggle/yolo26n_seg/train_kaggle.py` (`nahilr/coral-btp-yolo-seg`): Trains 7-class YOLO26n-seg with connected-component instance polygons (mIoU 0.4104).

Push to Kaggle via:
```bash
kaggle kernels push -p kaggle/yolo26n_sem/
kaggle kernels push -p kaggle/yolo26n_seg/
```

---

## Repository Layout

```
.
├── configs/
│   ├── labels.yaml                     # Canonical 7-class project taxonomy
│   ├── label_map.json                  # Explicit 95 -> 7 raw class mapping
│   └── dinov3_id2label.json            # 95-class ID-to-name mapping for DINOv3
├── kaggle/
│   ├── yolo26n_sem/                    # Kaggle kernel for YOLO26n-sem training
│   ├── yolo26n_seg/                    # Kaggle kernel for YOLO26n-seg training
│   └── README.md                       # Kaggle execution notes
├── models/
│   ├── checkpoints/
│   │   ├── yolo26n_sem/weights/best.pt # Fine-tuned 7-class YOLO26n-sem (Git LFS)
│   │   └── yolo26n_seg/weights/best.pt # Fine-tuned 7-class YOLO26n-seg (Git LFS)
│   └── pretrained/                     # SegFormer & DINOv3 backbones (via fetch_pretrained.py)
├── reports/
│   ├── model_comparison.md             # Benchmark summary across all 4 backends
│   └── model_comparison.json           # Machine-readable per-class IoU and confusion matrices
├── scripts/
│   ├── evaluate_model_comparison.py    # Reproducible 4-backend evaluator
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
│   │   └── pipeline.py                 # Multi-backend CoralPipeline & video processor
│   └── trust/
│       └── reliability.py              # Review decision gate, ECE, entropy
├── tests/
│   ├── test_dense_detections.py        # Region bounding box and gate tests
│   ├── test_health_indicators.py       # Reef health verdict and condition cap tests
│   ├── test_label_map.py               # Taxonomy mapping integrity tests
│   ├── test_quality.py                 # Image quality gate and BGR consistency tests
│   ├── test_video_pipeline.py          # Video-to-video inference test
│   └── run_all.py                      # Standalone test runner
├── main.py                             # CLI entry point (prepare, evaluate, infer, app)
└── requirements.txt                    # Minimal python dependencies
```

---

## Glossary & Terminology

| Term | Definition |
|---|---|
| **LCC** | Live Coral Cover: percentage of assessable reef substrate occupied by living coral colonies. |
| **Condition Cap** | An ecological safety rule capping the reported health tier when bleaching or mortality exceeds disturbance thresholds. |
| **Trust Decision** | Transparent `accepted` or `needs_review` state computed from image quality and prediction uncertainty. |
| **Region Box** | Display bounding box derived from connected components of dense coral masks (not per-colony instance annotations). |
| **Macro mIoU** | Mean Intersection over Union averaged over all valid classes with non-zero union in the dataset split. |
