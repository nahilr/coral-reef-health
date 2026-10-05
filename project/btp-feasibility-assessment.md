# Feasibility - Trustworthy AI-Assisted Coral Reef Health Assessment

## Verdict

**Feasible in 14-18 weeks** as a general underwater-imagery health-assessment system, provided the scope is restricted to (1) benthic composition, (2) visible bleaching/condition risk where labels exist, and (3) explicit reliability warnings. It is not feasible in the same period to claim disease diagnosis, a universal ecological-health score, real-world deployment accuracy everywhere, or causal environmental explanations.

## What the system can validly report

- Area proportion of broad benthic classes: coral, algae, rubble, sand/rock and other.
- Visible coral-condition probability: healthy-looking, bleached-looking, dead/rubble, or uncertain - only if trained/evaluated with those explicit labels.
- Image-quality, confidence and out-of-distribution (OOD) warnings.
- A survey/image-set summary that separates raw model coverage from accepted, high-confidence coverage.

## What must not be claimed

- Complete reef health, disease or resilience diagnosis.
- Coral species identification unless the training data and evaluation use that taxonomy.
- Bleaching from a generic coral segmentation model.
- Accurate predictions on a new reef, camera or water condition without a held-out evaluation or local validation.

## Feasibility by component

| Component | Feasible? | Notes |
|---|---:|---|
| Broad benthic semantic segmentation | Yes | Coralscapes has dense masks; this is the core model. |
| Visible condition/bleaching classification | Yes, conditional | Needs a dedicated labelled condition dataset and a clear label policy. |
| Image-quality filtering | Yes | Blur, brightness, contrast and color-cast checks are lightweight. |
| Calibration, OOD and abstention | Yes | Use held-out sites/sources, temperature scaling, and a review threshold. |
| Upload-and-report web demo | Yes | Build after model evaluation; local Streamlit or Gradio is sufficient. |
| Video support | Yes, limited | Sample frames and aggregate results; do not call adjacent frames independent evidence. |
| Real-time edge deployment | Optional | Export the final nano model to ONNX only after accuracy is stable. |

## Timeline

| Weeks | Outcome |
|---|---|
| 1-2 | Dataset cards, licences, label map, grouped train/validation/test protocol, reproducible environment. |
| 3-4 | Data conversion, image-quality audit, class-balance report and visual annotation checks. |
| 5-7 | Baseline semantic-segmentation experiment and error analysis. |
| 8-10 | Main YOLO26 semantic-segmentation model, ablation, held-out-site/source result. |
| 11-12 | Dedicated coral-condition classifier and fused indicator rules. |
| 13-14 | Calibration, OOD, abstention, failure gallery and survey-level aggregation. |
| 15-16 | Upload/report demo, packaging, final report and presentation. |
| 17-18 | Buffer for annotation correction, video demonstration or ONNX export. |

## Compute decision

Use **Google Colab for training** and any reasonable local machine for data preparation, inference and the demo. A local NVIDIA GPU with 8 GB VRAM can run the nano models; 12-16 GB is comfortable for 640-pixel experiments. CPU-only training is technically possible but not practical. Colab's T4-class GPU is sufficient for the core, using nano/small models and 512-640 px images.

## Delivery rule

The project is complete when it produces an evaluated segmentation model, an evaluated condition classifier or clearly documented condition-data limitation, reliability/review outputs, and a reproducible upload-to-report demo. A dashboard alone is not a completion criterion.
