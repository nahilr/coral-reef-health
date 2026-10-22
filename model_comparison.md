# Coralscapes Benchmark: Model Comparison

## Evaluation

All models used the same **473 images** from the official Coralscapes test split. The images were 2048×1024 pixels. Scores use the project's **7-class** taxonomy (`configs/labels.yaml`).

The evaluator maps ground-truth masks by class name (`src/data/label_map.py`) and leaves out void ground-truth pixels. It maps dense model outputs from Coralscapes names to the project classes. It sums the matching class probabilities before it selects the highest score. A void or unmapped prediction counts as an error. Macro mIoU is the mean IoU across classes that appear in the ground truth.

## Headline results

| Model | Task | Macro mIoU | Pixel accuracy | Inference | Weights |
|---|---|---:|---:|---:|---:|
| **DINOv3 ViT-L + LoRA + DPT** | semantic | **72.65%** | **87.20%** | 3.58 s/image | 1.33 GB |
| SegFormer-B5 | semantic | 67.87% | 85.36% | 1.30 s/image | 339 MB |
| YOLO26s-sem | semantic | 57.98% | 79.97% | 0.038 s/image² | 13.2 MB |
| YOLO26n-sem | semantic | 51.94% | 75.64% | 0.026 s/image¹ | 3.4 MB |
| YOLO26n-seg | instance → rasterised | 41.04% | 52.32% | 0.032 s/image | 6.6 MB |

¹ Single-image steady-state inference at `imgsz=1024`, batch 1, FP32 on NVIDIA GPU (~38.2 FPS).
² 473-image sequential benchmark at `imgsz=1024`, batch 1, FP32 on NVIDIA GeForce GTX 1650 (~26.1 FPS), excluding model loading.

YOLO26s-sem accuracy and per-class IoU are from its [50-epoch Kaggle run](https://www.kaggle.com/code/nahilr/coral-btp-yolo-sem-s); its speed was timed locally on the same official test images. The JSON report includes its confusion matrix and health-output scores as well.

DINOv3 scores **4.78 points** higher than SegFormer on mIoU and **1.84 points** higher on pixel accuracy. It takes 2.75 times longer to run. YOLO26s-sem scores **6.03 points** higher than YOLO26n-sem on mIoU and **4.34 points** higher on pixel accuracy. It runs at about 26 FPS on the GTX 1650.

## Per-class IoU

| Project class | YOLO26s-sem | YOLO26n-sem | YOLO26n-seg | SegFormer-B5 | DINOv3-DPT |
|---|---:|---:|---:|---:|---:|
| coral_alive | 65.1% | 59.5% | 62.0% | 75.1% | **80.5%** |
| coral_bleached | 57.0% | 49.9% | 36.7% | 71.6% | **83.6%** |
| coral_dead | 21.4% | 12.1% | 11.0% | 34.6% | **37.5%** |
| algae | 51.0% | 45.3% | 18.8% | 58.7% | **65.3%** |
| hard_substrate | 61.3% | 52.8% | 28.4% | 70.3% | **71.6%** |
| sand | 70.1% | 67.5% | 69.0% | 80.5% | **84.4%** |
| non_reef | 79.9% | 76.5% | 61.4% | 84.3% | **85.7%** |

All models score well on `coral_alive`, `coral_bleached`, `sand`, `hard_substrate`, and `non_reef`. The label map combines fine-grained algae labels into `algae` and substrate labels into `hard_substrate`.

## Reef-health agreement

This comparison uses the **464 test images with assessable reef area**. Live-cover MAE and bias are measured in percentage points. Tier agreement compares the pipeline verdict with the verdict from the ground-truth mask.

| Backend | Live-cover MAE | Live-cover bias | Health-tier agreement | Within one tier |
|---|---:|---:|---:|---:|
| **DINOv3 ViT-L + DPT** | **2.99 pts** | **+0.03 pts** | **80.5%** | **98.5%** |
| SegFormer-B5 | 3.87 pts | +1.17 pts | 77.8% | 97.5% |
| YOLO26s-sem | 5.23 pts | +3.31 pts | 70.0% | 96.8% |
| YOLO26n-sem | 7.16 pts | +2.35 pts | 64.5% | 93.2% |
| YOLO26n-seg | 23.34 pts | +22.67 pts | 47.8% | 81.0% |

## YOLO training runs

| Model | Run status | Settings | Training time | Result |
|---|---|---|---:|---|
| YOLO26n-sem | Complete | 30 epochs; 1024px; batch 4 | 2h 53m 06s | Test: 51.94% mIoU; 75.64% pixel accuracy |
| YOLO26n-seg | Complete | 30 epochs; 1024px; batch 4 | 1h 50m 19s | Test: 41.04% mIoU; 52.32% pixel accuracy; mask mAP50 26.05% |
| YOLO26s-sem | Complete | 50 epochs; 1024px; batch 4 | 5h 11m 32s training; 5h 22m 33s full Kaggle run | Test: 57.98% mIoU; 79.97% pixel accuracy |
| YOLO26m-sem | One-epoch smoke run | 1 epoch; 1024px; batch 4 | 6m 50s training; 17m 05s full Kaggle run | Validation only: 30.71% mIoU; 65.35% pixel accuracy |

Training times come from each run's Ultralytics `results.csv`. Full-run time includes setup and data conversion. It also includes official test evaluation for the 50-epoch YOLO26s-sem run. The YOLO26m-sem score is from one validation run. Do not compare it with test scores. Its prepared 85-epoch run has not been run. SegFormer-B5 and DINOv3-DPT use weights from their authors. They were not trained in these Kaggle runs.

## Why semantic and instance scores differ

Semantic segmentation assigns a class to each pixel. It fits continuous regions such as sand, rock, and water. Instance segmentation predicts separate boxes and masks. This works best for countable objects with clear boundaries.

This project creates instance masks from connected components in semantic labels (`kaggle/yolo26n_seg/train_kaggle.py`). This process turns large, continuous regions into artificial objects. The instance model can leave pixels unassigned when a detection does not pass its confidence and mask thresholds. Dense scoring counts those pixels as missed predictions. For these reasons, use semantic models to measure benthic cover and health ratios. Use instance masks to inspect individual regions, but do not use them for validated colony counts. The training labels are not expert colony-instance annotations.

## Protocols

- **YOLO26n-sem** — one aspect-preserving Ultralytics pass per image, `imgsz=1024`, FP32, dense 7-class `semantic_mask`, no confidence threshold.
- **YOLO26s-sem** — same aspect-preserving Ultralytics protocol, `imgsz=1024`, batch 1, FP32; timing measured over all 473 images on a GTX 1650.
- **YOLO26n-seg** — one pass at `imgsz=1024`, `conf=0.25`, FP32. Instance masks rasterised to a dense map with highest-confidence-wins. Uncovered pixels count as missed predictions.
- **SegFormer-B5** — model-card single pass at 768×1376 (direct resize), FP32, then nearest-neighbour resize of the project map to source dimensions.
- **DINOv3 ViT-L + DPT** — same 768×1376 single-pass protocol, FP32. Runs at 2.76 GB peak, so it fits a 4 GB card.

## Implementation notes

- **FP16 autocast is unusable for SegFormer here.** It returns NaN logits on this GPU and collapses every pixel to class 0. Everything runs in FP32; the corrected SegFormer pass runs at 1.30 s/image.
- **DINOv3 loads without the gated encoder.** `src/inference/dinov3_dpt.py` rebuilds the ViT config from the checkpoint's own tensor shapes (1024-d, 24 layers, 16 heads, MLP 4096, patch 16, 4 registers) and uses `AutoModel.from_config`, so no gated token is required.
- **Timings were measured during each backend's dedicated inference pass** in this evaluation session.

## Files

- Machine-readable metrics: `models/model_comparison.json`
- Reproducible five-model evaluator: `scripts/evaluate_model_comparison.py` (`--models yolo,yolo_s,yolo_seg,segformer,dinov3`, `--device`)
- Checkpoints: `models/checkpoints/yolo26s_sem`, `models/checkpoints/yolo26n_sem`, `models/checkpoints/yolo26n_seg`, `models/pretrained/segformer-b5-coralscapesv2-95`, `models/pretrained/coralscapesv2-dinov3-vitl-lora-dpt-95_class`
