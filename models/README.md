# Models

Fine-tuned checkpoints in `checkpoints/` are stored with **Git LFS** (run
`git lfs install` once per machine). The two public pretrained backbones in
`pretrained/` (~1.6 GB) are git-ignored and downloaded from Hugging Face.

## Checkpoints

| Path | Source | Notes |
|---|---|---|
| `checkpoints/yolo26n_seg/weights/best.pt` | Kaggle `nahilr/coral-btp-train` | V1 7-class instance segmentation baseline |
| `checkpoints/condition_resnet18.pt` | Local training | V1 crop condition classifier |
| `checkpoints/yolo26n_sem_v2/weights/best.pt` | Kaggle `nahilr/coral-btp-yolo-sem` | Native 7-class project taxonomy, 30 epochs. Test mIoU 0.5194, pixel accuracy 0.7564. |
| `checkpoints/yolo26n_seg_v2/weights/best.pt` | Kaggle `nahilr/coral-btp-yolo-seg` | Native 7-class instance segmentation on connected-component instances. Test mask mAP50 0.2605, mAP50-95 0.1355, mIoU 0.4104, pixel accuracy 0.5232. |

## Performance Summary

| Model | Architecture | Task | Macro mIoU | Pixel Acc | Mask mAP50 | Latency | Weights |
|---|---|---|---:|---:|---:|---:|---:|
| **DINOv3-DPT** | ViT-L + LoRA + DPT | Semantic (95→7 fold) | **72.65%** | **87.20%** | — | 3.58 s/img | 1.33 GB |
| **SegFormer-B5** | Hierarchical Transformer | Semantic (95→7 fold) | 67.87% | 85.36% | — | 1.30 s/img | 339 MB |
| **YOLO26n-sem (V2)** | Ultralytics YOLO26n-sem | Semantic (Native 7-class) | 51.94% | 75.64% | — | **0.026 s (~38 FPS)** | 3.4 MB |
| **YOLO26n-seg (V2)** | Ultralytics YOLO26n-seg | Instance (Native 7-class) | 41.04% | 52.32% | **26.05%** | 0.032 s (~31 FPS) | 6.6 MB |
| **YOLO26n-seg (V1)** | Ultralytics YOLO26n-seg | Instance (V1 baseline) | 40.38% | 74.37% | 16.52% | ~0.025 s (~40 FPS) | 6.5 MB |

### Instance (`seg`) vs. Semantic (`sem`) Models

- **`yolo26n_sem_v2` (Semantic)**: Classifies every pixel into the 7 project classes. Best for benthic coverage metrics (Live Coral Cover %, Bleaching %, Gomez category) and real-time streaming video (~38 FPS).
- **`yolo26n_seg_v2` (Instance)**: Delineates individual coral colonies with bounding boxes and polygon masks. Best for colony counting and size-frequency distributions (26.05% Mask mAP50).
- See [`README.md`](../README.md#instance-segmentation-seg-vs-semantic-segmentation-sem) for full conceptual, architectural, and ecological comparison.

## Pretrained

| Path | Source |
|---|---|
| `pretrained/segformer-b5-coralscapesv2-95` | Hugging Face `EPFL-ECEO/segformer-b5-finetuned-coralscapesv2-1024-1024-95_class` |
| `pretrained/coralscapesv2-dinov3-vitl-lora-dpt-95_class` | Hugging Face `EPFL-ECEO/coralscapesv2-dinov3-vitl-lora-dpt-95_class`. The encoder architecture is rebuilt from the checkpoint's own tensor shapes, so the gated `facebook/dinov3-vitl16-pretrain-lvd1689m` repo is not needed. `id2label.json` is copied in locally because the repo ships no label names. |

## Fresh clone

```bash
git lfs install                      # once per machine
git clone git@github.com:nahilr/coral-reef-health.git
cd coral-reef-health
python scripts/fetch_pretrained.py   # SegFormer + DINOv3 into models/pretrained/

python main.py infer --image sample.jpg \
  --model models/checkpoints/yolo26n_sem_v2/weights/best.pt --model-type yolo
```

`git lfs install` is per machine. Without it, the tracked `.pt` files arrive as
small pointer files instead of weights. A docs-only checkout can skip them:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:nahilr/coral-reef-health.git
```

Only `weights/best.pt` is versioned; `weights/last.pt` is a training-resume
artefact and is git-ignored. The public backbones can also be re-downloaded from
the Kaggle kernel outputs; see `kaggle/README.md` for kernel IDs and
`data/README.md` for dataset revision pins.
