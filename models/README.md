# Models

Fine-tuned YOLO checkpoints in `checkpoints/` are tracked with Git LFS. Install Git LFS before cloning or pulling the weights. SegFormer-B5 and DINOv3-DPT are public pretrained backbones; download them with `scripts/fetch_pretrained.py`.

## Current checkpoints

| Path | Source | Task | Test mIoU | Pixel accuracy | Additional metric |
|---|---|---|---:|---:|---:|
| `checkpoints/yolo26n_sem/weights/best.pt` | Kaggle `nahilr/coral-btp-yolo-sem` | 7-class semantic segmentation | 51.94% | 75.64% | 0.026 s/image |
| `checkpoints/yolo26n_seg/weights/best.pt` | Kaggle `nahilr/coral-btp-yolo-seg` | 7-class instance segmentation | 41.04% | 52.32% | mask mAP50 26.05% |

The YOLO results are from the 473-image CoralscapesV2 test split using the project taxonomy. The instance masks are trained from connected components derived from semantic annotations; they should not be treated as expert colony-instance annotations or used for validated colony counts.

## Four-backend benchmark

The same test split and project taxonomy are used for all four backends. Full metrics, per-class IoU, confusion matrices, protocols, and model-size/timing details are in [`reports/model_comparison.md`](../reports/model_comparison.md) and [`reports/model_comparison.json`](../reports/model_comparison.json).

| Model | Task | Test mIoU | Pixel accuracy | Inference |
|---|---|---:|---:|---:|
| DINOv3 ViT-L + LoRA + DPT | Semantic | 72.65% | 87.20% | 3.58 s/image |
| SegFormer-B5 | Semantic | 67.87% | 85.36% | 1.30 s/image |
| YOLO26n-sem | Semantic | 51.94% | 75.64% | 0.026 s/image |
| YOLO26n-seg | Instance, rasterized for dense scoring | 41.04% | 52.32% | 0.032 s/image |

## Pretrained backbones

| Local path | Source |
|---|---|
| `pretrained/segformer-b5-coralscapesv2-95` | `EPFL-ECEO/segformer-b5-finetuned-coralscapesv2-1024-1024-95_class` |
| `pretrained/coralscapesv2-dinov3-vitl-lora-dpt-95_class` | `EPFL-ECEO/coralscapesv2-dinov3-vitl-lora-dpt-95_class` |

The DINOv3 encoder configuration is rebuilt from checkpoint tensor shapes, so the gated base encoder repo is not needed. Its local model folder also needs `id2label.json` for label projection.

## Fresh clone

```bash
git lfs install
git clone git@github.com:nahilr/coral-reef-health.git
cd coral-reef-health
python scripts/fetch_pretrained.py
python main.py infer --image sample.jpg \
  --model models/checkpoints/yolo26n_sem/weights/best.pt --model-type yolo
```

Only `weights/best.pt` is versioned; `weights/last.pt` is ignored. A docs-only checkout can skip LFS smudging:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:nahilr/coral-reef-health.git
```
