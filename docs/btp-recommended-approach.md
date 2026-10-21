# End-to-End BTP Plan

## Product definition

Input: one underwater image or a short video. Output: benthic-area overlay, visible coral-condition estimate where supported, quality/reliability warnings, and a compact health-indicator report.

The system is **AI-assisted**: low-quality, low-confidence and out-of-distribution inputs are routed to human review rather than treated as facts.

## System architecture

```text
Image / video upload
  -> file + metadata validation
  -> frame sampling (video only) + quality gate
  -> semantic segmentation (7 mapped CoralscapesV2 project classes, including visible coral condition)
  -> reliability layer (image quality + backend-specific signals + review decision)
  -> aggregation (area proportions, accepted-frame coverage, condition summary)
  -> annotated image/video frames + JSON/CSV/PDF-ready report
```

## Datasets

| Role | Dataset | Use | Do not use it for |
|---|---|---|---|
| Primary segmentation | CoralscapesV2 | Train the 7 project classes from official dense masks; retain the original 95 labels and explicit mapping. | Disease diagnosis or a universal ecological health score. |
| Robustness | Official held-out V2 test split, then an external CoralNet/ReefNet/MERMAID source if rights and labels permit | Measure performance across held-out V2 sites and later across another source. | Random patch/frame split that leaks nearby imagery. |
| Condition labels | CoralscapesV2 visible-condition classes | Predict alive/bleached/dead only where source labels explicitly support the state; do not infer an unsupported condition. | Assume an unlabelled coral taxon is healthy. |
| Demo | Separate, never-before-seen images/video | Examples and qualitative review. | Training or threshold tuning. |

## Algorithms

| Stage | Choice | Why |
|---|---|---|
| Quality gate | Laplacian blur score, exposure/contrast checks, color-cast flag | Cheap, explainable and protects the model from unusable frames. |
| Segmentation backends | YOLO26n-sem, YOLO26n-seg, SegFormer-B5, and DINOv3-DPT using the shared 7-class project taxonomy | The current benchmark compares all four backends on the same CoralscapesV2 test split; results are recorded in `reports/model_comparison.md`. |
| Separate condition classifier | Not part of the consolidated workflow | The legacy V1 crop classifier was removed; visible condition labels are represented in the CoralscapesV2 segmentation taxonomy where explicitly annotated. |
| Trust layer | Image-quality checks and backend-specific confidence or uncertainty signals, with an explicit review decision | Calibration and OOD detection are future evaluation work; do not report them as completed measurements. |

## Training protocol

1. Freeze the 7-class V2 dictionary in `configs/labels.yaml`. Map all 95 source labels explicitly in `configs/label_map.json`; report reef metrics over reef classes only and route scene content (water, fish, humans, transect hardware) to the `non_reef` discard sink.
2. Split by **dive site/source/image group**, not pixels, crops or neighboring video frames. Keep a sealed final test set.
3. The Kaggle YOLO26n-sem training workflow uses 1024 px inputs, batch size 4, mixed precision and early stopping. Tune resolution and batch size using validation results, not the sealed test split.
4. Apply realistic augmentations: horizontal flip, moderate brightness/contrast/hue shift, blur and compression. Do not use transformations that invent bleaching or erase biological color evidence.
5. Address class imbalance with class-aware sampling/loss weighting and report every class separately.
6. Select hyperparameters on validation only. Keep the final test split sealed until evaluation.
7. Evaluate each implemented segmentation backend with mIoU, per-class IoU, pixel accuracy and visual error maps. Report calibration or OOD metrics only when those evaluations have actually been run.
8. Report the official CoralscapesV2 test-split results and state clearly that they do not establish performance on other reefs, cameras, or survey conditions.

## Indicator rules

- Benthic composition = accepted segmentation pixels by mapped class / accepted valid pixels.
- Coral-condition distribution = accepted pixels from explicit alive/bleached/dead classes; show its denominator and unknown-condition share.
- “Needs review” if input fails quality checks, available backend signals indicate an unreliable prediction, or coral coverage is too small for a reliable condition estimate.
- Display `visible condition indicators`, not a single definitive health score.

## Minimum viable deliverables

1. Dataset cards, licences, label map and fixed split manifest.
2. Released V2 SegFormer baseline and the project’s mapped YOLO result, evaluated on the held-out V2 test split.
3. Documented quality/reliability flags and backend comparison results.
4. Local upload demo that exports overlay plus machine-readable report.

## Scope and trust contract

The feasible deliverable is broad benthic composition, visible coral-condition indicators only where explicit labels support them, and clear quality/reliability warnings. Do not claim disease diagnosis, species identification without matching labels, a universal reef-health score, or validated accuracy on a new reef/camera without held-out or local evaluation. Video support in the current scope means fixed-rate frame sampling, deduplication where possible, and aggregation with frame counts/coverage; it does not mean colony tracking, 3D reconstruction, metric transect mapping, or repeat-survey change detection.

Use semantic segmentation for area/cover estimates. Detection boxes do not directly measure class area. For CoralscapesV2, visible condition is part of the mapped semantic output only where labels explicitly support the state; the legacy V1 crop classifier has been removed. Treat enhancement as an optional measured experiment and preserve the original pixels. Reliability is a required gate: every result should record model/data version, quality flags, backend confidence or uncertainty when available, accepted-pixel counts, and an explicit `needs review` outcome.

## Feasibility and staged delivery

| Stage | Indicative weeks | Outcome |
|---|---:|---|
| Data and protocol | 1–2 | Dataset cards/rights, label map, grouped split manifest, reproducible environment. |
| Audit and baseline | 3–7 | Data conversion/quality audit, class balance and visual checks, baseline segmentation and error analysis. |
| Model comparison and validation | 8–10 | Evaluate the four implemented backends on the held-out CoralscapesV2 test split. |
| Reliability and health indicators | 11–14 | Integrate image-quality checks, backend-specific signals, review flags, and supported condition indicators. |
| Integration and report | 15–16 | Upload/report demo, packaging, final report and presentation. Weeks 17–18 are buffer. |

This schedule is feasible for a bounded prototype in roughly 14–18 weeks, conditional on usable labels and compute. Where V2 does not provide an explicit health state, preserve the unknown-condition label rather than infer a state. Completion requires evaluated segmentation, reliability/review outputs, and a reproducible upload-to-report workflow; a dashboard alone is insufficient.

## Environment and reproducibility

- Use Python 3.10/3.11, PyTorch matched to the runtime, Ultralytics, OpenCV, NumPy, pandas, scikit-learn, and pinned dependencies. Keep reusable training/evaluation code in versioned Python files and notebooks small.
- Colab is the default training environment; local machines can handle data inspection, inference, and the demo. An NVIDIA GPU with 8 GB VRAM can run nano-model experiments; 12–16 GB is more comfortable. Start at 512 px and scale to 640 only when held-out gains justify the cost. CPU-only training is not practical for the core runs.
- Record data version, split manifest, configuration, random seed, package/runtime versions, GPU details, checkpoints, and metrics for every run. Never tune against the sealed test set.
- Keep raw inputs immutable; store converted masks/crops and manifests separately; keep large checkpoints out of Git. Review at least 20 successes and 20 failures, including low-quality or shifted inputs, before finalizing claims.
