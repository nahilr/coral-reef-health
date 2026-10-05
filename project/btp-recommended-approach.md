# End-to-End BTP Plan

## Product definition

Input: one underwater image or a short video. Output: benthic-area overlay, visible coral-condition estimate where supported, quality/reliability warnings, and a compact health-indicator report.

The system is **AI-assisted**: low-quality, low-confidence and out-of-distribution inputs are routed to human review rather than treated as facts.

## System architecture

```text
Image / video upload
  -> file + metadata validation
  -> frame sampling (video only) + quality gate
  -> Model A: semantic segmentation (coral / algae / rubble / sand-rock / other)
  -> Model B: coral-crop condition classifier (healthy-looking / bleached-looking / dead-rubble / uncertain)
  -> reliability layer (calibration + OOD + abstain/review)
  -> aggregation (area proportions, accepted-frame coverage, condition summary)
  -> annotated image/video frames + JSON/CSV/PDF-ready report
```

## Datasets

| Role | Dataset | Use | Do not use it for |
|---|---|---|---|
| Model A primary | Coralscapes | Train/validate broad benthic semantic segmentation. Collapse the 39 labels only through a published label map. | Direct bleaching or disease prediction. |
| Model A robustness | A held-out Coralscapes site/country; then one external CoralNet/ReefNet/MERMAID source if licence permits | Honest domain-shift test. | Random patch split that leaks location/camera information. |
| Model B primary | NOAA PIFSC coral-bleaching dataset, or a manually audited set of coral crops with healthy/bleached/dead labels | Train a visible-condition classifier. | Fine-grained species or disease claims. |
| Model B fallback | Annotate 300-500 diverse coral crops from legitimately usable imagery with a written expert label guide | Proof-of-concept condition model. | A final scientific bleaching claim without expert review. |
| Demo | Separate, never-before-seen images/video | Examples and qualitative review. | Training or threshold tuning. |

## Algorithms

| Stage | Choice | Why |
|---|---|---|
| Quality gate | Laplacian blur score, exposure/contrast checks, color-cast flag | Cheap, explainable and protects the model from unusable frames. |
| Segmentation baseline | DeepLabV3+ with ResNet-50 encoder or SegFormer-B0 | Gives a credible non-YOLO comparison. |
| Segmentation main model | YOLO26 semantic segmentation (`yolo26n-sem`, then `yolo26s-sem` only if justified) | Current, deployable task-specific model family. |
| Condition baseline | EfficientNet-B0 or ResNet-18 classifier on coral crops | Small and easy to calibrate. |
| Condition main model | EfficientNet-B2 or a small ViT, fine-tuned from ImageNet weights | A single stronger comparison, not a model zoo. |
| Trust layer | Temperature scaling, ECE/reliability diagram, maximum-softmax OOD baseline, quality/OOD/confidence abstention rule | Turns uncertainty into a reviewable decision. |

## Training protocol

1. Freeze a label dictionary and map every source label to `coral`, `algae`, `rubble`, `sand_rock`, `other`, or `unmapped`. Preserve the original label.
2. Split by **dive site/source/image group**, not pixels, crops or neighboring video frames. Keep a sealed final test set.
3. Begin at 512 px, then use 640 px only if it improves held-out metrics and fits memory. Use pretrained weights, mixed precision and early stopping.
4. Apply realistic augmentations: horizontal flip, moderate brightness/contrast/hue shift, blur and compression. Do not use transformations that invent bleaching or erase biological color evidence.
5. Address class imbalance with class-aware sampling/loss weighting and report every class separately.
6. Select hyperparameters on validation only. Calibrate after the model is frozen. Choose the abstention threshold based on a target accepted-case accuracy/coverage tradeoff.
7. Evaluate Model A with mIoU, per-class IoU, pixel accuracy and visual error maps. Evaluate Model B with macro-F1, balanced accuracy, per-class precision/recall, confusion matrix and calibration metrics.
8. Run a held-out-site/source evaluation. Publish both in-domain and shifted-domain results; the gap is part of the result.

## Indicator rules

- Benthic composition = accepted segmentation pixels by broad class / accepted valid pixels.
- Coral-condition distribution = accepted coral crops only; show its denominator.
- “Needs review” if input fails quality, OOD is high, confidence is below threshold, or coral coverage is too small for a reliable condition estimate.
- Display `visible condition indicators`, not a single definitive health score.

## Minimum viable deliverables

1. Dataset cards, licences, label map and fixed split manifest.
2. Model A baseline and YOLO26 result with held-out-site evaluation.
3. Model B result or a documented, evidence-based reason it is deferred.
4. Calibration/OOD/abstention report and failure gallery.
5. Local upload demo that exports overlay plus machine-readable report.
