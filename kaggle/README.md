# Kaggle Jobs

One folder per model version. Each folder is an immutable snapshot of the code that was pushed to Kaggle.

| Folder | Kernel ID | Model | Status |
|---|---|---|---|
| `yolo26n_seg_v1/` | `nahilr/coral-btp-train` | YOLO26n-seg, 7-class instance seg | Superseded by V2 |
| `yolo26n_sem_v2/` | `nahilr/coral-btp-yolo-sem` | YOLO26n-sem, 20-class semantic | Current |
| `yolo26n_seg_pseudo_v2/` | `nahilr/coral-btp-yolo-seg-v2` | YOLO26n-seg, pseudo-instances from V2 masks | Experiment |

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
