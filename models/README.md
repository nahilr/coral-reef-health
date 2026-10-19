# Models

Model weights are git-ignored. This README documents what exists and how to obtain it.

## Checkpoints

| Path | Source | Notes |
|---|---|---|
| `checkpoints/yolo26n_seg/weights/best.pt` | Kaggle `nahilr/coral-btp-train` | V1 7-class instance segmentation baseline |
| `checkpoints/condition_resnet18.pt` | Local training | V1 crop condition classifier |
| `checkpoints/yolo26n_sem_v2/weights/best.pt` | Kaggle `nahilr/coral-btp-yolo-sem` | V2 20-class semantic segmentation |

## Pretrained

| Path | Source |
|---|---|
| `pretrained/segformer-b5-coralscapesv2-95` | Hugging Face `mstrahl/segformer-b5-coralscapesv2-95` |

## Downloading

Weights can be re-downloaded from Kaggle kernel outputs or Hugging Face. See `kaggle/README.md` for kernel IDs and `data/README.md` for dataset revision pins.
