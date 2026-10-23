# Models

Fine-tuned YOLO checkpoints in `checkpoints/` are tracked with Git LFS. Install Git LFS before cloning or pulling the weights. SegFormer-B5 and DINOv3-DPT are public pretrained backbones; download them with `scripts/fetch_pretrained.py`.

## Current checkpoints

| Path | Source | Task | Test mIoU | Pixel accuracy | Inference on GTX 1650 | Weights |
|---|---|---|---:|---:|---:|---:|
| `checkpoints/yolo26m_sem/weights/best.pt` | Kaggle `nahilr/coral-btp-yolo26m-85ep` | 7-class semantic segmentation | **60.23%**¹ | **81.13%**¹ | 0.111 s/image | 28.9 MB |
| `checkpoints/yolo26s_sem/weights/best.pt` | Kaggle `nahilr/coral-btp-yolo-sem-s` | 7-class semantic segmentation | 57.98% | 79.97% | 0.038 s/image | 13.2 MB |
| `checkpoints/yolo26n_sem/weights/best.pt` | Kaggle `nahilr/coral-btp-yolo-sem` | 7-class semantic segmentation | 51.94% | 75.64% | 0.026 s/image | 3.4 MB |
| `checkpoints/yolo26n_seg/weights/best.pt` | Kaggle `nahilr/coral-btp-yolo-seg` | 7-class instance segmentation | 41.04% | 52.32% | 0.032 s/image | 6.6 MB |

¹ Local replay on the original PNG test images. Kaggle's run report measured 60.17% mIoU and 81.06% pixel accuracy after its test images were converted to JPEG. All scores cover the same 473-image official split. For per-class results, training times, protocols, and the full comparison, see the [model comparison report](../model_comparison.md). The instance masks come from connected components in semantic annotations. They are not expert colony-instance labels and do not support validated colony counts.

The YOLO26m-sem folder also includes its training arguments, results CSV, training and validation previews, and benchmark plots alongside `best.pt`, matching the retained artifacts for the other YOLO runs.

The machine-readable benchmark data and confusion matrices are in [`model_comparison.json`](model_comparison.json).

## Pretrained backbones

| Local path | Source |
|---|---|
| `pretrained/segformer-b5-coralscapesv2-95` | `EPFL-ECEO/segformer-b5-finetuned-coralscapesv2-1024-1024-95_class` |
| `pretrained/coralscapesv2-dinov3-vitl-lora-dpt-95_class` | `EPFL-ECEO/coralscapesv2-dinov3-vitl-lora-dpt-95_class` |

The DINOv3 loader reads the encoder shape from the checkpoint. It does not need the gated base encoder repo. Add `id2label.json` to the local model folder so the loader can map labels.

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
