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
| Segmentation backends | YOLO26n-sem, YOLO26s-sem, YOLO26n-seg, SegFormer-B5, and DINOv3-DPT using the shared 7-class project taxonomy | The benchmark compares all five backends on the same CoralscapesV2 test split; results are recorded in [`model_comparison.md`](../model_comparison.md). |
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

### YOLO26n-sem accuracy improvement plan

The current YOLO26n-sem checkpoint reaches **51.94% 7-class mIoU** and **75.64% pixel accuracy** on the 473-image official test split. Its local training run used 30 epochs, 1024-pixel images, and batch size 4. The best validation mIoU was **0.4854 at epoch 26**; the run ended at **0.4839**. This suggests that longer training is worth testing, but may not be enough on its own.

The lowest test IoU is `coral_dead` at **12.1%**. `coral_bleached` scores **49.9%**, `algae` **45.3%**, and `hard_substrate` **52.8%**. Before changing sampling or loss weights, inspect the 7-class confusion matrix, training-pixel counts, and masks for these classes.

The CoralscapesV2 authors report the following results. These scores use different class maps from this project, so they are context, not direct targets.

| Published model | Classes | Pixel accuracy | mIoU |
|---|---:|---:|---:|
| SegFormer MiT-B5 | 39 | 82.801% | 59.020% |
| DINOv3 ViT-L + LoRA + DPT | 39 | 84.390% | 66.086% |
| SegFormer MiT-B5 | 95 | 80.382% | 37.177% |
| DINOv3 ViT-L + LoRA + DPT | 95 | 81.700% | 40.221% |

The authors trained SegFormer for 100 epochs on train and validation data with AdamW, a learning rate of `6e-5`, and weight decay of `0.01`. They used random crops, color changes, rotations, and flips. Their DINOv3 model uses a frozen pretrained ViT with LoRA and a DPT head. They trained it for 150 epochs with multi-scale sizes and crop and scale augmentations. The 7-class project score cannot be compared directly with these 39- or 95-class scores. Merging source classes changes the metric.

Use this experiment order:

| Priority | Experiment | Method |
|---:|---|---|
| 1 | Check data and evaluation | Compare source images, converted JPEGs, and masks. Check image-mask alignment, color conversion, class mapping, ignored pixels, resize behavior, and the 7×7 confusion matrix. Run the current and candidate checkpoints through the same evaluator. |
| 2 | Train longer | Compare 60–100 epochs with patience of 20–30. This is a low-cost experiment; the current run had nearly plateaued by epoch 26, so longer training may give only a small gain. Save the best validation-mIoU checkpoint and inspect full learning curves. Select settings on validation. Retrain on train plus validation with a fixed schedule, then evaluate once on the sealed test split. |
| 3 | Compare model sizes | Compare YOLO26n-sem, YOLO26s-sem, and YOLO26m-sem where GPU memory allows. Keep other settings fixed. |
| 4 | Tune resolution and augmentation | Compare 1024 with 1280 or 1536 pixels if memory allows. Test tiled crops if full-resolution training is too costly. Reduce color-distorting HSV changes, taper mosaic near the end, and avoid erasing that hides small or rare regions. Keep moderate scale changes and flips. |
| 5 | Address rare classes | If counts and confusion results confirm class imbalance, test class-weighted loss or rare-class crop sampling. Compare macro mIoU and per-class scores, and check for losses on common classes. |
| 6 | Distill from DINOv3 | If simpler changes reach a plateau, test teacher probabilities or pseudo-labels. Keep human labels authoritative and mask uncertain or void teacher pixels. |

Keep the official test split sealed during model selection. Report both the project 7-class scores and per-class results. Do not promise that YOLO26n-sem will reach the published transformer scores. A sound goal is to improve on the current **51.94% 7-class mIoU** and reduce the gap to SegFormer under the same project evaluation.

The V2 training workflow writes JPEG images at quality 95, while the comparison report describes evaluation on source images. Confirm image-mask alignment and document this preprocessing difference. The test evaluation should use the original images if that is the stated protocol.

Sources: [CoralscapesV2 project page and model results](https://josauder.github.io/coralscapesv2/), [CoralscapesV2 paper](https://arxiv.org/abs/2609.12826), [Ultralytics semantic segmentation guide](https://docs.ultralytics.com/tasks/semantic), and [YOLO26 semantic model configuration](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/26/yolo26-sem.yaml). Project scores and per-class results are in [`model_comparison.md`](../model_comparison.md) and [`models/model_comparison.json`](../models/model_comparison.json).

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
| Model comparison and validation | 8–10 | Evaluate the five implemented backends on the held-out CoralscapesV2 test split. |
| Reliability and health indicators | 11–14 | Integrate image-quality checks, backend-specific signals, review flags, and supported condition indicators. |
| Integration and report | 15–16 | Upload/report demo, packaging, final report and presentation. Weeks 17–18 are buffer. |

This schedule is feasible for a bounded prototype in roughly 14–18 weeks, conditional on usable labels and compute. Where V2 does not provide an explicit health state, preserve the unknown-condition label rather than infer a state. Completion requires evaluated segmentation, reliability/review outputs, and a reproducible upload-to-report workflow; a dashboard alone is insufficient.

## Environment and reproducibility

- Use Python 3.10/3.11, PyTorch matched to the runtime, Ultralytics, OpenCV, NumPy, pandas, scikit-learn, and pinned dependencies. Keep reusable training/evaluation code in versioned Python files and notebooks small.
- Colab is the default training environment; local machines can handle data inspection, inference, and the demo. An NVIDIA GPU with 8 GB VRAM can run nano-model experiments; 12–16 GB is more comfortable. Start at 512 px and scale to 640 only when held-out gains justify the cost. CPU-only training is not practical for the core runs.
- Record data version, split manifest, configuration, random seed, package/runtime versions, GPU details, checkpoints, and metrics for every run. Never tune against the sealed test set.
- Keep raw inputs immutable; store converted masks/crops and manifests separately; keep large checkpoints out of Git. Review at least 20 successes and 20 failures, including low-quality or shifted inputs, before finalizing claims.
