# Kaggle Jobs

GPU training scripts for the 7-class YOLO models on Kaggle.

| Folder | Kernel Slug | Model | Status |
|---|---|---|---|
| `yolo26n_sem/` | `nahilr/coral-btp-yolo-sem` | YOLO26n-sem, 7-class project taxonomy | Trained & verified (30 epochs, mIoU 0.5194) |
| `yolo26n_seg/` | `nahilr/coral-btp-yolo-seg` | YOLO26n-seg, 7-class project taxonomy | Trained & verified (30 epochs, mIoU 0.4104) |
| `yolo26s_sem/` | `nahilr/coral-btp-yolo-sem-s` | YOLO26s-sem, 7-class project taxonomy | Completed (50 epochs, 5h 11m training; test mIoU 0.5798) |
| `yolo26m_sem_smoke/` | `nahilr/coral-btp-yolo26m-smoke` | YOLO26m-sem, 7-class project taxonomy | Smoke run completed (1 epoch, 6m 50s training; validation mIoU 0.3071) |
| `yolo26m_sem/` | `nahilr/coral-btp-yolo26m-85ep` | YOLO26m-sem, 7-class project taxonomy | 85-epoch run prepared locally; not pushed |

The times are elapsed training durations from `results.csv`; the 50-epoch S Kaggle run took 5h 22m 33s end-to-end, while the M smoke kernel took 17m 05s including setup and data conversion. The M smoke metrics are validation-only and are not comparable with official test scores from completed runs.

All kernels emit the same 7-class taxonomy (`configs/labels.yaml`). A Kaggle kernel has no
repository copy, so each inlines the source-label map; keep those tables identical to
`configs/label_map.json`.

## Pushing to Kaggle

Push directly from the folder:
```bash
kaggle kernels push -p kaggle/yolo26n_sem/
kaggle kernels push -p kaggle/yolo26n_seg/
kaggle kernels push -p kaggle/yolo26s_sem/
kaggle kernels push -p kaggle/yolo26m_sem/
```

## Outputs

Download outputs via:
```bash
kaggle kernels output <user>/<slug> -p outputs/kaggle/<folder-name>/
```
