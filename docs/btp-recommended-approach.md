# End-to-End BTP Plan

## Product definition

Input: one underwater image or a short video. Output: benthic-area overlay, visible coral-condition estimate where supported, quality/reliability warnings, and a compact health-indicator report.

The system is **AI-assisted**: low-quality, low-confidence and out-of-distribution inputs are routed to human review rather than treated as facts.

## System architecture

```text
Image / video upload
  -> file + metadata validation
  -> frame sampling (video only) + quality gate
  -> semantic segmentation (20 mapped CoralscapesV2 classes, including visible coral condition)
  -> reliability layer (calibration + OOD + abstain/review)
  -> aggregation (area proportions, accepted-frame coverage, condition summary)
  -> annotated image/video frames + JSON/CSV/PDF-ready report
```

## Datasets

| Role | Dataset | Use | Do not use it for |
|---|---|---|---|
| Primary segmentation | CoralscapesV2 | Train 20 project classes from the official dense masks; retain the original 95 labels and the explicit mapping. | Disease diagnosis or a universal ecological health score. |
| Robustness | Official held-out V2 test split, then an external CoralNet/ReefNet/MERMAID source if rights and labels permit | Measure performance across held-out V2 sites and later across another source. | Random patch/frame split that leaks nearby imagery. |
| Condition labels | CoralscapesV2 visible-condition classes | Predict alive/bleached/dead where labels explicitly support it; retain an unknown-condition class. | Assume an unlabelled coral taxon is healthy. |
| Demo | Separate, never-before-seen images/video | Examples and qualitative review. | Training or threshold tuning. |

## Algorithms

| Stage | Choice | Why |
|---|---|---|
| Quality gate | Laplacian blur score, exposure/contrast checks, color-cast flag | Cheap, explainable and protects the model from unusable frames. |
| Segmentation benchmark | Released CoralscapesV2 SegFormer-B5 model with predictions mapped to project classes | Provides a reef-trained reference point using the dataset authors’ evaluation protocol. |
| Segmentation project model | YOLO26n-sem trained on dense 20-class PNG masks | Produces a full-scene class map suited to cover estimates; compare its held-out results with the released SegFormer benchmark. |
| Separate condition classifier | Not used by default for V2 | The old V1 crop classifier remains available for baseline comparison; V2 state labels are included in segmentation. |
| Trust layer | Temperature scaling, ECE/reliability diagram, maximum-softmax OOD baseline, quality/OOD/confidence abstention rule | Turns uncertainty into a reviewable decision. |

## Training protocol

1. Freeze the 20-class V2 dictionary in `configs/labels.yaml`. Preserve each original V2 label and map only labels with explicit support; retain `coral_condition_unknown`.
2. Split by **dive site/source/image group**, not pixels, crops or neighboring video frames. Keep a sealed final test set.
3. The initial Kaggle run uses pretrained YOLO26n-sem at 1024 px with batch size 4, mixed precision and early stopping. Tune resolution and batch size using validation results, not the sealed test split.
4. Apply realistic augmentations: horizontal flip, moderate brightness/contrast/hue shift, blur and compression. Do not use transformations that invent bleaching or erase biological color evidence.
5. Address class imbalance with class-aware sampling/loss weighting and report every class separately.
6. Select hyperparameters on validation only. Calibrate after the model is frozen. Choose the abstention threshold based on a target accepted-case accuracy/coverage tradeoff.
7. Evaluate Model A with mIoU, per-class IoU, pixel accuracy and visual error maps. Evaluate Model B with macro-F1, balanced accuracy, per-class precision/recall, confusion matrix and calibration metrics.
8. Run a held-out-site/source evaluation. Publish both in-domain and shifted-domain results; the gap is part of the result.

## Indicator rules

- Benthic composition = accepted segmentation pixels by mapped class / accepted valid pixels.
- Coral-condition distribution = accepted pixels from explicit alive/bleached/dead classes; show its denominator and unknown-condition share.
- “Needs review” if input fails quality, OOD is high, confidence is below threshold, or coral coverage is too small for a reliable condition estimate.
- Display `visible condition indicators`, not a single definitive health score.

## Minimum viable deliverables

1. Dataset cards, licences, label map and fixed split manifest.
2. Released V2 SegFormer baseline and the project’s mapped YOLO result, evaluated on the held-out V2 test split.
3. Calibration/OOD/abstention report and failure gallery.
4. Local upload demo that exports overlay plus machine-readable report.

## Scope and trust contract

The feasible deliverable is broad benthic composition, visible coral-condition indicators only where explicit labels support them, and clear quality/reliability warnings. Do not claim disease diagnosis, species identification without matching labels, a universal reef-health score, or validated accuracy on a new reef/camera without held-out or local evaluation. Video support in the current scope means fixed-rate frame sampling, deduplication where possible, and aggregation with frame counts/coverage; it does not mean colony tracking, 3D reconstruction, metric transect mapping, or repeat-survey change detection.

Use semantic segmentation for area/cover estimates. Detection boxes do not directly measure class area. For V2, condition is part of the mapped semantic output only where labels explicitly support the state; do not load the weak V1 crop classifier by default. Treat enhancement as an optional measured experiment and preserve the original pixels. Reliability is a required gate: every result should record model/data version, quality flags, confidence or uncertainty state, accepted-pixel counts, and an explicit `needs review` outcome.

## Feasibility and staged delivery

| Stage | Indicative weeks | Outcome |
|---|---:|---|
| Data and protocol | 1–2 | Dataset cards/rights, label map, grouped split manifest, reproducible environment. |
| Audit and baseline | 3–7 | Data conversion/quality audit, class balance and visual checks, baseline segmentation and error analysis. |
| Main model and validation | 8–10 | YOLO26 segmentation, ablation, held-out-site/source result. |
| Reliability and health indicators | 11–14 | Calibration, OOD/abstention evaluation, failure gallery, and aggregation of supported condition labels. |
| Integration and report | 15–16 | Upload/report demo, packaging, final report and presentation. Weeks 17–18 are buffer. |

This schedule is feasible for a bounded prototype in roughly 14–18 weeks, conditional on usable labels and compute. Where V2 does not provide an explicit health state, preserve the unknown-condition label rather than infer a state. Completion requires evaluated segmentation, reliability/review outputs, and a reproducible upload-to-report workflow; a dashboard alone is insufficient.

## Environment and reproducibility

- Use Python 3.10/3.11, PyTorch matched to the runtime, Ultralytics, OpenCV, NumPy, pandas, scikit-learn, and pinned dependencies. Keep reusable training/evaluation code in versioned Python files and notebooks small.
- Colab is the default training environment; local machines can handle data inspection, inference, and the demo. An NVIDIA GPU with 8 GB VRAM can run nano-model experiments; 12–16 GB is more comfortable. Start at 512 px and scale to 640 only when held-out gains justify the cost. CPU-only training is not practical for the core runs.
- Record data version, split manifest, configuration, random seed, package/runtime versions, GPU details, checkpoints, and metrics for every run. Never tune against the sealed test set.
- Keep raw inputs immutable; store converted masks/crops and manifests separately; keep large checkpoints out of Git. Review at least 20 successes and 20 failures, including low-quality or shifted inputs, before finalizing claims.
