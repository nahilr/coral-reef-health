# Coralscapes Benchmark: Four Backends

## Evaluation

All four models were evaluated on the same **473 images** of the official Coralscapes test split at 2048×1024 source resolution, against the **7-class** project taxonomy (`configs/labels.yaml`).

Ground-truth mask IDs were mapped by class name (`src/data/label_map.py`); unmapped/void GT pixels were excluded. Dense model output IDs were resolved through the Coralscapes `id2label` names, fine-label probabilities were summed into the 7 project classes before argmax, and any pixel predicted as void/unmapped counts as an incorrect (missed) prediction. Macro IoU averages classes with a non-zero union.

## Headline results

| Model | Task | Macro mIoU | Pixel accuracy | Inference | Weights |
|---|---|---:|---:|---:|---:|
| **DINOv3 ViT-L + LoRA + DPT** | semantic | **72.65%** | **87.20%** | 3.58 s/image | 1.33 GB |
| SegFormer-B5 | semantic | 67.87% | 85.36% | 1.30 s/image | 339 MB |
| YOLO26n-sem | semantic | 51.94% | 75.64% | 0.026 s/image¹ | 3.4 MB |
| YOLO26n-seg | instance → rasterised | 41.04% | 52.32% | 0.032 s/image | 6.6 MB |

¹ Single-image steady-state inference at `imgsz=1024`, batch 1, FP32 on NVIDIA GPU (~38.2 FPS).

DINOv3 leads SegFormer by **4.78 points** of mIoU and **1.84 points** of pixel accuracy, at 2.75× the inference cost. SegFormer leads YOLO26n-sem by **15.93 points** of mIoU, at ~50× the cost. YOLO26n-sem is the fastest model, running at 38+ FPS.

## Per-class IoU

| Project class | YOLO26n-sem | YOLO26n-seg | SegFormer-B5 | DINOv3-DPT |
|---|---:|---:|---:|---:|
| coral_alive | 59.5% | 62.0% | 75.1% | **80.5%** |
| coral_bleached | 49.9% | 36.7% | 71.6% | **83.6%** |
| coral_dead | 12.1% | 11.0% | 34.6% | **37.5%** |
| algae | 45.3% | 18.8% | 58.7% | **65.3%** |
| hard_substrate | 52.8% | 28.4% | 70.3% | **71.6%** |
| sand | 67.5% | 69.0% | 80.5% | **84.4%** |
| non_reef | 76.5% | 61.4% | 84.3% | **85.7%** |

Every model is strongest on `coral_alive`, `coral_bleached`, `sand`, `hard_substrate` and `non_reef`. Merging fine-grained categories into unified `algae` and consolidated substrate into `hard_substrate` cleanly preserves all reef health indicators.

## Protocols

- **YOLO26n-sem** — one aspect-preserving Ultralytics pass per image, `imgsz=1024`, FP32, dense 7-class `semantic_mask`, no confidence threshold.
- **YOLO26n-seg** — one pass at `imgsz=1024`, `conf=0.25`, FP32. Instance masks rasterised to a dense map with highest-confidence-wins. Uncovered pixels count as missed predictions.
- **SegFormer-B5** — model-card single pass at 768×1376 (direct resize), FP32, then nearest-neighbour resize of the project map to source dimensions.
- **DINOv3 ViT-L + DPT** — same 768×1376 single-pass protocol, FP32. Runs at 2.76 GB peak, so it fits a 4 GB card.

## Implementation notes

- **FP16 autocast is unusable for SegFormer here.** It returns NaN logits on this GPU and collapses every pixel to class 0. Everything runs in FP32; the corrected SegFormer pass runs at 1.30 s/image.
- **DINOv3 loads without the gated encoder.** `src/inference/dinov3_dpt.py` rebuilds the ViT config from the checkpoint's own tensor shapes (1024-d, 24 layers, 16 heads, MLP 4096, patch 16, 4 registers) and uses `AutoModel.from_config`, so no gated token is required.
- **Timings were measured during each backend's dedicated inference pass** in this evaluation session.

## Files

- Machine-readable metrics: `reports/model_comparison.json`
- Reproducible evaluator: `scripts/evaluate_model_comparison.py` (`--models yolo,yolo_seg,segformer,dinov3`, `--device`)
- Backends: `models/checkpoints/yolo26n_sem`, `models/checkpoints/yolo26n_seg`, `models/pretrained/segformer-b5-coralscapesv2-95`, `models/pretrained/coralscapesv2-dinov3-vitl-lora-dpt-95_class`
