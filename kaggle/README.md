# Kaggle Jobs

GPU training scripts for the 7-class YOLO models on Kaggle.

| Folder | Kernel Slug | Model | Status |
|---|---|---|---|
| `yolo26n_sem/` | `nahilr/coral-btp-yolo-sem` | YOLO26n-sem, 7-class project taxonomy | Trained & verified (30 epochs, mIoU 0.5194) |
| `yolo26n_seg/` | `nahilr/coral-btp-yolo-seg` | YOLO26n-seg, 7-class project taxonomy | Trained & verified (30 epochs, mIoU 0.4104) |

Both kernels emit the same 7-class taxonomy (`configs/labels.yaml`). A Kaggle kernel has no
repository copy, so each inlines the source-label map; keep those tables identical to
`configs/label_map.json`.

## Pushing to Kaggle

Push directly from the folder:
```bash
kaggle kernels push -p kaggle/yolo26n_sem/
kaggle kernels push -p kaggle/yolo26n_seg/
```

## Outputs

Download outputs via:
```bash
kaggle kernels output <user>/<slug> -p outputs/kaggle/<folder-name>/
```
