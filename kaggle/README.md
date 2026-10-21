# Kaggle Jobs

One folder per model version. Each folder is an immutable snapshot of the code that was pushed to Kaggle.

| Folder | Kernel ID | Model | Status |
|---|---|---|---|
| `yolo26n_seg_v1/` | `nahilr/coral-btp-train` | YOLO26n-seg, 7-class instance seg (CoralscapesV1) | Historical baseline |
| `yolo26n_sem_v2/` | `nahilr/coral-btp-yolo-sem` | YOLO26n-sem, 7-class project taxonomy | Trained & verified (30 epochs, mIoU 0.5194) |
| `yolo26n_seg_v2/` | `nahilr/coral-btp-yolo-seg` | YOLO26n-seg, 7-class project taxonomy, instances derived from semantic masks | Trained & verified (30 epochs, mIoU 0.4104) |

Both V2 kernels emit the same 7-class taxonomy (`configs/labels.yaml`). A Kaggle kernel has no
repository copy, so each inlines the source-label map; keep those tables identical to
`configs/label_map.json`. The `*_v2` folder and kernel names are the existing Kaggle slugs and are
kept as-is so pushes reuse them.

## Pushing a new version

1. Copy the current kernel folder to a new folder (e.g. `yolo26n_sem_v3/`).
2. Edit code inside the new folder.
3. Push from that folder:
   ```bash
   kaggle kernels push -p kaggle/yolo26n_sem_v3/
   ```
4. Update this README.

## Outputs

Download outputs to `outputs/kaggle/<folder-name>/`. Promote metrics to `reports/`.
