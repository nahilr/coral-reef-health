# Coral Reef Health Assessment — Full Project Study Guide

A beginner-friendly guide to the problem, computer-vision and machine-learning concepts, project data, models, pipeline, results, limitations, and presentation questions.

> **One-sentence summary:** Given an underwater reef image or sampled video frames, the project labels every visible pixel into one of seven benthic/scene categories, turns those pixel counts into visible-cover indicators, flags questionable inputs or predictions for expert review, and saves visual and JSON outputs.

## 1. What problem does the project address?

Coral reefs are complex ecosystems. Researchers monitor them by surveying underwater scenes and estimating which organisms or substrates occupy the seafloor. Useful indicators include the fraction of observed bottom covered by living coral, bleached coral, dead coral, algae, rock/rubble, and sand. Manual annotation is informative but slow and expensive, especially across many images or repeated surveys.

This project explores an AI-assisted workflow for making those observations faster and more repeatable. It is a **computer-vision measurement aid**, not an autonomous marine biologist. It predicts visual categories from pixels. It does not directly measure water temperature, coral growth, biodiversity, disease, reef resilience, or whole-ecosystem health.

The image-derived categories can support monitoring when the camera view is representative, labels are valid for the location, predictions are checked, and the resulting measurements are interpreted alongside field and environmental information.

### What “health” means here

The app reports **visible cover** and **coral condition** as separate indicators:

- **Live coral cover (LCC):** pixels predicted as `coral_alive` as a share of predicted reef/benthic pixels. In the current implementation, `coral_bleached` pixels are reported separately and are not included in LCC.
- **Bleaching ratio:** predicted bleached coral pixels as a share of predicted coral pixels (alive + bleached + dead).
- **Mortality ratio:** predicted dead coral pixels as a share of predicted coral pixels.
- **Algae cover:** predicted algae pixels as a share of predicted reef/benthic pixels.

Coral condition uses bleaching and dead-coral ratios among detected coral only. If no coral is detected, condition is **Not assessable (No coral detected)**; a sand-only scene can still report 0% LCC and a **Poor** cover tier. The app displays condition and cover separately. JSON `reef_health_category` remains a combined cover-and-condition summary.

The cover tier uses the labels **Excellent**, **Good**, **Fair**, and **Poor**. “Poor / Degraded” was shortened to “Poor”; the 10% LCC cutoff did not change. The JSON combined category can still be capped downward when bleaching or the dead-coral ratio is high.

These names can sound broader than the underlying measurement. In a presentation, call the result a **visible-cover / condition indicator** or **image-derived health indicator**, and explain that local ecological validation is still needed.

## 2. Computer-vision basics

### Image as a grid of pixels

A digital image is a rectangular grid. Each pixel stores color values, commonly red, green, and blue (RGB), each usually from 0 to 255. Underwater cameras may produce color casts, haze, low contrast, blur, uneven lighting, and suspended-particle noise. These effects can hide visual cues and make the same coral appear different from its training examples.

A video is a time-ordered sequence of image frames. A frame is just a still image with a timestamp/index; analyzing frames independently does not automatically create object tracking or understand motion.

### Classification, detection, and segmentation

- **Image classification:** assign one or more labels to a whole image, such as “reef” or “bleaching event.” It does not say where the object is.
- **Object detection:** return boxes around discrete objects, usually with a class and confidence score. Good for counting separated objects; a box includes background pixels.
- **Semantic segmentation:** assign a category to every pixel. Adjacent regions of the same class share a label, even if they are separate organisms. This suits area composition and cover estimation.
- **Instance segmentation:** assign a separate mask to each detected object. It distinguishes individual instances but expects object-like annotations and often leaves some pixels unassigned.

The main application needs **semantic segmentation** because area/cover is calculated from pixel classes. YOLO26n-seg is included as an instance-segmentation comparison; it is not the best representation for continuous sand, algae, or substrate regions.

### Segmentation output

For each pixel, a dense model usually produces a score (logit) for each class. A softmax converts scores into class probabilities that sum to one. The most probable class is selected by `argmax`. The resulting **class map** has one integer class ID per pixel. A **mask** is a binary map for one class: 1 means the pixel belongs to that category, 0 means it does not.

A model may also return confidence (largest class probability) per pixel. Confidence is not the same as correctness: a model can be confidently wrong, especially on unfamiliar reefs, cameras, lighting, or species.

### How a neural network “sees” an image

A **convolutional neural network (CNN)** applies small learned filters (kernels) across an image. Early filters respond to simple patterns such as edges and color transitions; deeper layers combine them into textures and shapes. As feature maps are downsampled, each unit sees a larger part of the image (its **receptive field**). Segmentation heads combine features at different scales and predict labels over a spatial grid. YOLO-family networks primarily use efficient local feature processing alongside task-specific prediction heads.

A **Transformer** divides an image into patches/tokens and uses **self-attention** to let each patch combine information from other patches. Attention is useful when a local region is ambiguous but broader context helps; it can also require more compute and memory. Hierarchical transformers such as SegFormer retain features at multiple resolutions. DINOv3 learns general visual representations through self-supervised pretraining, then its features are adapted by a decoder for dense output. Exact layer-by-layer details depend on each released checkpoint/configuration.

### Basic probability terms

- **Logit:** an unnormalized class score emitted by a model.
- **Probability:** a score transformed or normalized to a 0–1 scale; for multi-class softmax outputs, probabilities across classes sum to one.
- **Threshold:** a cutoff used to accept/reject a prediction or flag a condition. Changing it trades false positives against false negatives.
- **Entropy:** high when probability mass is spread across alternatives; low when one class dominates. It measures decisiveness, not whether the answer is right.

## 3. Machine-learning concepts needed to understand the system

### Training, validation, and test data

A supervised segmentation model learns from images paired with human-annotated masks:

1. **Training set:** used to adjust model parameters.
2. **Validation set:** used to choose training duration, model settings, and thresholds.
3. **Test set:** kept separate until final evaluation; estimates performance on held-out examples.

The project uses the official CoralscapesV2 split and reports results on its 473-image test set. A test score estimates performance on that benchmark distribution, not on every reef. In marine imaging, related frames from the same dive/site can be very similar. Splitting by image alone can leak scene information; site/source grouping is preferred when building new splits.

### Model parameters and learning

A neural network is a stack of trainable numerical operations. During training, it compares predictions to known labels with a **loss function**. Backpropagation computes how each parameter contributed to that error, and an optimizer (such as SGD or AdamW) updates parameters. An **epoch** is one pass through training data; **batch size** is the number of examples used for one update. **Image size/resolution** affects both detail and memory/speed.

A **pretrained model** starts with weights learned from an earlier, often much larger dataset. **Fine-tuning** adapts those weights to the target imagery and labels. Transfer learning is useful when target labels are limited, but it does not eliminate domain shift.

### Overfitting and generalization

A model overfits when it memorizes training-specific details and performs poorly on new scenes. Augmentation (e.g. flips, moderate color/exposure changes, scales) presents varied versions of training images, but transformations must preserve biological evidence. Excessive hue changes could make bleached and healthy coral look artificially alike.

**Domain shift** occurs when deployment images differ from training images: geography, species, depth, camera, lighting, water clarity, season, or annotation policy. A model performing well on CoralscapesV2 may still need local validation for Lakshadweep or another camera system.

### Why accuracy alone is not enough

Pixel accuracy is the fraction of valid pixels predicted correctly. It can look high when large/common classes dominate. For each class, **intersection over union (IoU)** is:

`IoU = true-positive pixels / (true-positive + false-positive + false-negative pixels)`

**Macro mIoU** averages class IoU values, giving rare and common classes equal class-level weight. This is more informative for imbalanced segmentation. Per-class IoU and a confusion matrix reveal which categories are confused. A confusion matrix counts true class versus predicted class.

Other metrics in this project:

- **MAE (mean absolute error):** average absolute difference between predicted and reference cover, measured in percentage points here.
- **Bias:** average signed error. Positive live-cover bias means overestimation on average.
- **Tier agreement:** fraction of images where the categorical tier matches the ground-truth-derived tier.
- **Confidence calibration:** whether predictions assigned, for example, 0.8 confidence are correct about 80% of the time across comparable predictions. Confidence alone is not calibration, and neither proves out-of-distribution safety.

## 4. Dataset and label design

### CoralscapesV2

The current training and five-model benchmark use **CoralscapesV2**, a dataset of 2,433 densely annotated reef images with 95 fine-grained labels and official train/validation/test splits. The project benchmark uses 473 official test images at 2048 × 1024 pixels. Training scripts stream the dataset from Hugging Face on Kaggle; the full V2 training corpus is not required to be stored locally for those runs.

The dataset’s original 95 labels include coral forms/condition states and other benthic or scene categories. The project merges these into a stable 7-class vocabulary. This reduces task complexity and produces meaningful cover categories, but it loses species/genus and morphology detail. Label merging changes the task and can change apparent metric scores.

### The seven project classes

| ID | Class | Meaning in this project |
|---:|---|---|
| 0 | `coral_alive` | Living coral labels mapped from the source taxonomy |
| 1 | `coral_bleached` | Visibly bleached coral where the source label explicitly says bleached |
| 2 | `coral_dead` | Dead coral labels where the source taxonomy supports that state |
| 3 | `algae` | Algae and algae-covered substrate/coral source labels |
| 4 | `hard_substrate` | Hard bottom, rubble, and related consolidated substrate |
| 5 | `sand` | Sand and soft sediment |
| 6 | `non_reef` | Scene content outside the reef-cover calculation, such as water, fish, people, or transect gear |

The exact fine-label-to-project-label mapping is in `configs/label_map.json`. Unmapped/void labels use ignore ID 255 in the mask conversion/evaluation path. Mapping by explicit class name makes the taxonomy reviewable and avoids silently assuming that a class ID means the same thing in every model.

### Reef denominator and excluded scene pixels

The six classes 0–5 represent the project’s benthic/reef denominator. `non_reef` is predicted for the overlay/report but excluded from cover calculations so a large patch of open water or a transect marker does not lower coral cover. This is an operational denominator choice, not a universal ecological convention.

### Preparing data for YOLO

The optional local preparation path reads image and mask bytes from Parquet shards, maps the original labels to the seven project labels, then extracts external contours for each class. It approximates contours as polygons, removes very small contours, normalizes polygon coordinates to the image width/height, and writes YOLO-format labels plus `data.yaml` and `manifest.csv`.

This is a conversion from dense semantic masks to polygon objects. Thin or tiny regions may be lost through area filtering or contour simplification. For YOLO semantic models, the actual training behavior is defined by the Ultralytics semantic model/task code; for YOLO instance-segmentation, the project’s instance polygons are derived from connected components of semantic labels, not hand-labeled coral colonies. Therefore instance detections must not be presented as validated colony counts.

## 5. Models in this project

The five backends share a common evaluation taxonomy and output interface, but differ in architecture, capacity, training origin, speed, and output type.

### YOLO26n-sem and YOLO26s-sem

These are Ultralytics YOLO26 **semantic segmentation** models fine-tuned for the project’s seven classes. YOLO combines convolutional feature extraction with multi-scale prediction and a segmentation output head. Its “n” and “s” sizes indicate smaller/larger model variants: the larger model generally uses more computation and memory but can learn richer features. The `-sem` suffix means dense semantic class maps, not per-colony instances.

- YOLO26n-sem: 30 epochs, 1024-pixel training size, batch 4; compact checkpoint (~3.4 MB), fastest among benchmarked semantic models.
- YOLO26s-sem: 50 epochs, 1024-pixel training size, batch 4; larger checkpoint (~13.2 MB) and better benchmark scores than the nano variant.

Training runs were performed on Kaggle GPUs. Inference speed numbers in the report were measured on an NVIDIA GTX 1650 and exclude model-loading time; do not generalize that speed to every device.

### YOLO26n-seg

This is an Ultralytics **instance segmentation** model. It outputs detected masks, class IDs, bounding boxes, and detection confidence. The pipeline paints those masks into one dense map, letting higher-confidence masks claim overlapping pixels. Pixels left uncovered count as missed in the dense benchmark. Its training labels are connected components extracted from semantic masks, so its “instances” are annotation fragments/regions rather than verified biological colonies. It is useful for demonstrating the distinction between instance and semantic tasks, but is a poor choice for quantitative whole-bottom cover here.

### SegFormer-B5

SegFormer is a transformer-based semantic segmentation architecture. Its MiT (Mix Transformer) encoder extracts features at multiple spatial resolutions; a lightweight decoder combines those features into per-pixel classes. B5 is a relatively large model variant. The project uses the released CoralscapesV2 95-class checkpoint, then folds fine-class probabilities into the seven project classes by summing probabilities for all source labels mapped to the same target class before choosing a class.

This is important: if source labels “brain coral alive” and “massive alive” both map to `coral_alive`, their probabilities are combined. Simply taking the highest fine label first could produce a different project class decision.

### DINOv3 ViT-L + LoRA + DPT

This backend combines three ideas:

- **DINOv3:** a vision-transformer image encoder pretrained with self-supervised learning. It produces rich visual representations without needing a class label for every pretraining image.
- **ViT-L:** the “large” Vision Transformer encoder variant; it divides the image into patches, mixes information across patches with self-attention, and produces feature representations.
- **LoRA:** Low-Rank Adaptation, a parameter-efficient way to adapt a large model by learning small low-rank updates instead of changing every base parameter. The released checkpoint has its adapters merged for inference.
- **DPT:** Dense Prediction Transformer decoder/head, which turns transformer features into a spatial dense segmentation output.

The project uses an EPFL-ECEO CoralscapesV2 95-class checkpoint. Its loader reconstructs the DINOv3 encoder configuration from checkpoint tensor shapes, so it can load without downloading the gated base encoder. The weights are much larger (~1.33 GB), and inference is slower than YOLO, but this backend scored highest in the project’s test comparison.

### Why five models?

The comparison illustrates an accuracy/compute trade-off and tests distinct design choices: small real-time semantic models, a larger semantic YOLO, an instance-segmentation model, a hierarchical transformer, and a large pretrained vision transformer with a dense decoder. It does not prove one architecture is universally best. The best deployment choice depends on the purpose: quick edge inference, cover accuracy, available GPU memory, or expert-review workflow.

## 6. End-to-end workflow and pipeline

### Training/evaluation workflow

1. Retrieve CoralscapesV2 images, masks, and label metadata.
2. Map fine-grained source labels to the seven project categories.
3. For YOLO training, convert class masks to polygon labels and configure the train/validation/test folders.
4. Train YOLO checkpoints on Kaggle. SegFormer-B5 and DINOv3-DPT are author-provided CoralscapesV2 checkpoints rather than models trained in the project’s Kaggle runs.
5. Run all five backends on the same 473-image official test split.
6. Map model predictions into the project labels and compute pixel accuracy, per-class IoU, macro mIoU, and derived cover-indicator agreement.

### Single image inference

`main.py infer` uses `CoralPipeline`:

1. **Read the image** using OpenCV (internal arrays are BGR).
2. **Assess image quality:** calculate Laplacian variance as a blur indicator, grayscale mean as exposure, grayscale standard deviation as contrast, and a channel-mean ratio as a simple color-cast warning.
3. **Run a selected model:** YOLO semantic, YOLO instance, SegFormer, or DINOv3-DPT.
4. **Convert to a dense class map:** resize predictions back to the source image dimensions; for instance masks, resolve overlaps by confidence.
5. **Count pixels per class** and calculate class proportions.
6. **Compute health indicators** from the pixel counts.
7. **Apply the review gate:** combine image-quality flags with available confidence, uncertainty, and predicted reef coverage.
8. **Render an overlay:** blend class colors over the source image, draw coral-region boxes, and display a small telemetry banner.
9. **Save outputs:** overlay image and `report.json` with model information, quality, composition, health indicators, detections, and reliability status.

Quality scores are inexpensive heuristics. A color cast may be caused by underwater optics but is not itself proof of a bad image; blur/exposure/contrast thresholds are not universal. This gate warns; it does not restore colors or reject the image from inference.

### Video inference

For a video, OpenCV reads frames and the pipeline processes every Nth frame (and optionally stops after a maximum). Each sampled frame is run through the same image pipeline. The app can show frame progress and save an annotated MP4 plus a JSON report containing per-frame reports and survey aggregates.

The current video workflow is **fixed-rate sampling plus frame-wise aggregation**. It does not track the same coral colony across time, de-duplicate near-identical views robustly, reconstruct 3D reef geometry, compute absolute surveyed area, or perform repeat-survey change detection. Adjacent video frames are correlated, so the number of sampled frames is not the number of independent reef observations.

## 7. Health-indicator formulas and tier logic

Let `total_valid` be the number of pixels with a recognized predicted class, and `non_reef` the predicted scene pixels excluded from reef calculations:

`reef_area_pixels = total_valid - non_reef_pixels`

Then:

- `LCC (%) = coral_alive_pixels / reef_area_pixels × 100`
- `Bleached cover (%) = coral_bleached_pixels / reef_area_pixels × 100`
- `Dead cover (%) = coral_dead_pixels / reef_area_pixels × 100`
- `Algae cover (%) = algae_pixels / reef_area_pixels × 100`
- `Bleaching ratio (%) = coral_bleached_pixels / (alive + bleached + dead pixels) × 100`
- `Dead-coral ratio (%) = coral_dead_pixels / (alive + bleached + dead pixels) × 100` (the JSON field is still named `mortality_ratio_pct`)
- `Coral-to-algae ratio = alive pixels / algae pixels` (0 when algae count is zero)

Example: if the valid reef denominator has 10,000 pixels, including 3,000 living coral pixels, LCC is 30%. If coral pixels total 4,000 and 400 are bleached, the bleaching ratio is 10%.

### Cover tier

The implementation uses these LCC bands:

- 50% or more: Excellent
- 25% to under 50%: Good
- 10% to under 25%: Fair
- under 10%: Poor

The JSON combined summary applies a downward cap for bleaching or the dead-coral ratio: a ratio of at least 10% caps the cover tier at Good; at least 25% caps it at Fair; at least 50% caps it at Poor. The app's Coral Condition display is separate and does not use sand or other substrate cover to assign coral condition. The underlying percentage formulas and condition thresholds were not changed as part of the display updates. These are project rules; validate them against the monitoring protocol before treating them as universal standards.

Existing JSON reports and rendered overlays in `data/processed` are saved outputs and are not rewritten when source labels change. Re-run inference to regenerate them with the current display labels and no-coral condition handling.

## 8. Reliability and human review

The project attempts to make uncertainty visible through three kinds of signals:

1. **Input quality:** blur, exposure, contrast, color cast, unreadable input.
2. **Prediction confidence:** mean maximum pixel probability for dense models, or detection confidence for instance predictions when available.
3. **Prediction uncertainty:** normalized entropy of class probabilities for dense backends, or entropy derived from detection confidence for the instance backend.
4. **Coverage:** fraction of the image predicted as benthic/reef classes, excluding `non_reef`.

The review gate returns `accepted` or `needs_review` and records reasons. Thresholds differ by backend; transformer defaults are stricter in the pipeline. If a backend does not expose confidence/uncertainty, the code skips that check instead of fabricating a value.

This is **a heuristic review gate, not a validated out-of-distribution detector or guarantee of correctness**. A model can be confidently wrong. A proper calibration study needs held-out probability/correctness analysis; an OOD study needs known in-distribution and shifted data with suitable metrics. The review flag is a cue to inspect outputs, not a certification.

## 9. Benchmark results and how to explain them

All reported models were evaluated on 473 official CoralscapesV2 test images using the project’s seven-class taxonomy. Headline values from `model_comparison.md`:

| Model | Task | Macro mIoU | Pixel accuracy | Reported inference time/image |
|---|---|---:|---:|---:|
| DINOv3 ViT-L + LoRA + DPT | Semantic | **72.65%** | **87.20%** | 3.58 s |
| SegFormer-B5 | Semantic | 67.87% | 85.36% | 1.30 s |
| YOLO26s-sem | Semantic | 57.98% | 79.97% | 0.038 s |
| YOLO26n-sem | Semantic | 51.94% | 75.64% | 0.026 s |
| YOLO26n-seg | Instance, rasterized to dense map | 41.04% | 52.32% | 0.032 s |

YOLO timing was measured on an NVIDIA GTX 1650 and excludes model loading. Transformer timing also comes from a dedicated evaluation pass; hardware/timing differences mean these are approximate project measurements, not a hardware-independent speed ranking. DINOv3 is the most accurate in this comparison but much larger/slower. YOLO26n-sem is much smaller and faster. YOLO26s-sem improves accuracy over nano while remaining compact.

### Per-class story

DINOv3 leads on most classes, including `coral_alive` (80.5% IoU), `coral_bleached` (83.6%), sand (84.4%), and non-reef (85.7%). `coral_dead` is the hardest class for every model: best reported IoU is 37.5% for DINOv3. Dead coral has visual ambiguity and may have limited/uneven examples. This is why overall score is not the whole story; rare or important classes need their own error review.

### Derived cover results

On 464 test images with assessable reef area, DINOv3’s live-cover MAE was 2.99 percentage points and its mean bias was +0.03 points; its health-tier agreement was 80.5%. SegFormer’s live-cover MAE was 3.87 points. The YOLO26n-seg live-cover MAE was 23.34 points, with +22.67-point bias, consistent with large unassigned regions in the rasterized instance output.

### Fair comparison caveats

- The class taxonomy is merged to seven classes; these are **not** the authors’ original 95-class scores.
- Semantic and instance outputs behave differently; YOLO26n-seg is penalized when it leaves dense pixels uncovered.
- Backend preprocessing/resizing differs: Ultralytics preserves aspect ratio at a specified size; SegFormer and DINOv3 use direct resize to 1376 × 768, then restore the project map. Resize can alter shape/detail.
- The evaluation represents CoralscapesV2’s test distribution, not local field validation.
- mIoU, pixel accuracy, instance mask mAP, object tracking metrics, survey repeatability, and cover MAE measure different tasks and must not be ranked as one common metric.

## 10. Research background and how it relates

The research landscape includes several distinct approaches:

- **Dense benthic segmentation** predicts composition over the whole image and most directly supports pixel-derived cover. CoralscapesV2 is the current model/data anchor.
- **Point-label and patch-classification systems** such as CoralNet scale benthic composition surveys, but sample points/patches rather than necessarily producing full masks.
- **Colony detection and tracking** systems such as YOLO + DeepSORT follow discrete coral objects through video. This project’s video code does not do tracking.
- **Photogrammetry and 3D approaches** estimate surface area/geometry and can study before/after change, but this project does not reconstruct 3D reef surfaces.
- **Condition classifiers** predict coral condition or stress labels, often at image/patch level; their outputs and evaluation metrics differ from pixel segmentation.
- **Satellite habitat monitoring** supports larger-area and long-term trends, but uses different sensors/resolution and cannot substitute for underwater close-up labels.

The most useful related studies answer different questions:

| Research | What it contributes | Connection and difference from this project |
|---|---|---|
| **Coralscapes / CoralscapesV2** | Dense reef-scene semantic labels and a common benchmark; V2 expands to 95 fine-grained classes and adds video-context fish instances. | This is the current source dataset and pretrained-transformer task. The project remaps its labels into seven broader categories. ([V1 paper](https://openaccess.thecvf.com/content/ICCV2025W/CVAUI%20%26%20AAMVEM/papers/Sauder_The_Coralscapes_Dataset_Semantic_Scene_Understanding_in_Coral_Reefs_ICCVW_2025_paper.pdf), [V2 project/paper](https://josauder.github.io/coralscapesv2/)) |
| **González-Rivero et al. (2020)** | AI-based benthic image analysis aggregated into abundance/composition and compared with experts across bioregions. | Closest precedent for translating image predictions into ecological cover summaries; this project has not yet replicated their expert agreement/error validation. ([paper](https://www.mdpi.com/2072-4292/12/3/489)) |
| **Sauder et al. (2025), DeepReefMap** | Converts reef video into 3D semantic maps and survey-scale cover estimates, with repeated-survey analysis. | Closest broad video-to-cover workflow. This project samples frames and averages frame estimates; it does not create 3D maps, normalize transects geometrically, or report repeat-survey fidelity. ([paper](https://www.nature.com/articles/s41598-025-20795-z)) |
| **Zheng et al. (2025), CoralVOS** | Dense coral video object segmentation benchmark across many annotated frames and sites. | Relevant to temporally consistent video masks, but the project predicts benthic classes independently on selected frames; CoralVOS J&F is not comparable to project mIoU. ([paper](https://openaccess.thecvf.com/content/ICCV2025W/CVAUI%20%26%20AAMVEM/papers/Zheng_CoralVOS_Dataset_and_Benchmark_for_Dense_Coral_Video_Segmentation_ICCVW_2025_paper.pdf)) |
| **Ouassine et al. (2025), YOLOv8 + DeepSORT** | Detects coral objects and links identities through video. | This was named as a proposal-stage baseline. The current code does not track colony identities; its object-detection/tracking metrics answer another question. ([DOI](https://doi.org/10.1016/j.ecoinf.2025.103170)) |
| **Kopecky et al. (2023)** | Combines underwater photogrammetry and AI-assisted segmentation to quantify live/dead coral change around a bleaching event. | Supports the scientific value of segmentation for condition outcomes; their 3D surface area and pixel classification measures are not directly comparable to this project’s 2D mIoU. ([paper](https://doi.org/10.3390/rs15164077)) |
| **Wyatt et al. (2025, 2026)** | Separately studies OOD detection for reef image surveys and uncertainty-aware abundance estimation. | Closest safety/reliability research. The project’s confidence/entropy review flags are not a completed OOD benchmark, calibrated uncertainty interval, or expert-validated abundance estimator. ([OOD study](https://www.sciencedirect.com/science/article/pii/S157495412500216X), [uncertainty study](https://doi.org/10.1016/j.ecoinf.2026.103831)) |
| **Shao et al. (2025/2026)** | Uses DINOv2 with adapter learning for multi-label coral condition monitoring across places/seasons. | Provides condition-recognition/generalization context; it is an image-level multi-label task, not seven-class per-pixel segmentation. ([DOI](https://doi.org/10.1016/j.marpolbul.2025.119054)) |

Other context includes CoralNet’s point/patch annotation workflow, Sea-thru underwater color restoration, and satellite habitat-change studies. These are not implemented as project components. The companion [`paper inventory`](paper-inventory.md) records additional citations, local PDF paths, reported metrics, and exact task differences.

The key research gap is not simply “get a larger model.” A deployable ecological instrument needs evaluation across held-out reefs/cameras, comparisons to expert area/cover measurements, repeated-survey consistency, calibration/uncertainty checks, and local validation. The present project establishes an end-to-end prototype and a controlled same-dataset model comparison; it does not complete those validations.

## 11. What is implemented versus proposed

### Implemented in the current code

- Seven-class label mapping and dense semantic inference.
- YOLO26n/s semantic and YOLO26n instance models, plus SegFormer-B5 and DINOv3-DPT backends.
- Image quality heuristics, confidence/entropy signals where available, and an accepted/needs-review gate.
- Pixel composition and derived visible-cover/condition indicators.
- Image overlay and JSON report.
- Video frame subsampling, per-frame inference, annotated MP4, and frame-level aggregation.
- Five-model evaluation on the 473-image official CoralscapesV2 test split.

### Not implemented or not validated as a project claim

- Lakshadweep-specific model training/validation.
- Colony identity tracking across frames, 3D reconstruction, area-calibrated transects, or repeat-survey change detection.
- Integration with NOAA heat stress, fish datasets, or proposed Moorea/CoralNet/Dryad historical data in the inference pipeline.
- Autonomous disease diagnosis, species-level identification, or a universal reef-health score.
- A full external OOD test, expert field validation of predicted cover, or assurance that confidence reflects correctness.

## 12. A presentation-ready explanation

### 30-second version

“We built an AI-assisted coral-reef image analysis pipeline. It takes an underwater photo or sampled video, checks basic image quality, and uses semantic segmentation to label each pixel as live, bleached, or dead coral, algae, hard substrate, sand, or non-reef scene content. We convert those labels into visible-cover indicators and flag uncertain or poor-quality results for human review. On the 473-image CoralscapesV2 test split, DINOv3-DPT reached 72.65% macro mIoU, while the much smaller YOLO26n-sem reached 51.94% and was far faster. These are benchmark results, not validation on every reef.”

### 2-minute version

“Manual reef surveys are valuable but annotating large collections of underwater imagery takes time. Our prototype automates part of that workflow. The model predicts a category for every pixel, not just a label for the whole image. We use CoralscapesV2, which has 95 detailed labels, and map those labels into seven broad classes so the outputs can support cover summaries. Non-reef scene pixels such as water and equipment are excluded from the benthic denominator.

“We compare five models: small and larger YOLO semantic segmenters, a YOLO instance segmenter, SegFormer-B5, and DINOv3 with a DPT decoder. Dense semantic predictions are the natural fit for area cover. Instance masks are useful for object-shaped detections, but our instance labels were generated from semantic regions, not expert colony identities. In the shared 473-image test benchmark, DINOv3 has the highest score at 72.65% mIoU; YOLO26n-sem is much faster and smaller at 51.94% mIoU. Dead coral is still difficult, and the benchmark does not establish performance on another geography or camera.

“After segmentation we count pixels, calculate live coral cover, bleaching and mortality ratios, and attach simple quality and uncertainty flags. We use these as review cues. The software samples video frames and averages frame summaries, but it does not track colonies or measure temporal change. The next scientific step is expert and local validation, plus tests on held-out sites and cameras.”

### Running a live demonstration

The repository’s basic commands are:

```bash
# One image with the default YOLO26n semantic backend
python main.py infer --image path/to/reef.jpg --out outputs/demo

# Sample video at every second frame
python main.py infer --video path/to/survey.mp4 --every-n 2 --out outputs/video_demo

# Launch the interactive dashboard
python main.py app --port 8501
```

The image/video command writes an overlay and a JSON report. The dashboard supports backend selection, uploads, visual inspection, and downloads. Model checkpoints and optional transformer weights must be present locally; a clean checkout may need Git LFS and the optional pretrained-weight fetch step. A good demo should use a known sample image and explain the overlay, report metrics, and review status together.

## 13. Likely questions and defensible answers

**Q: Why semantic segmentation instead of just detection?**
Area cover depends on how many pixels each class occupies. Boxes include background and instance masks are designed for separate objects; dense semantic masks fit continuous reef surfaces.

**Q: Why group 95 labels into 7?**
The project needs stable broad cover/condition categories and enough common task classes to compare models. Grouping reduces detail, so the output cannot answer fine species/genus questions.

**Q: Does a high confidence mean the result is correct?**
No. It is a model score. Reliability requires calibration and tests on new conditions. The current gate is a practical flag for review, not proof of correctness.

**Q: Why is DINOv3 better?**
On this dataset split and seven-class evaluation, the released DINOv3 ViT-L + LoRA + DPT checkpoint scored highest. Its pretrained transformer features are powerful, but the result does not prove universal superiority; it is larger, slower, and may behave differently under domain shift.

**Q: Why does YOLO26n-seg score lower?**
It predicts separate instances, then the project rasterizes detections into a dense map. Pixels with no retained detection are counted as misses. It also learns pseudo-instances created from semantic connected components rather than expert colony instances.

**Q: What does 72.65% mIoU mean?**
For each of the seven classes, IoU measures overlap between predicted and annotated pixels. mIoU is their average. It means average class overlap under this benchmark protocol—not “72.65% of coral health is correct.”

**Q: Does it work on Lakshadweep reefs?**
The current benchmark does not establish that. The model uses CoralscapesV2 data, so local images and expert annotations are needed to assess transfer and recalibrate or retrain if necessary.

**Q: Is the output a medical-style diagnosis of the reef?**
No. It summarizes visible cover/condition labels in an image. Broader reef health needs biological, environmental, spatial, temporal, and expert context.

**Q: What is the next step?**
Evaluate on independent sites and cameras; compare cover estimates with experts; inspect errors by class; validate bleaching/dead-coral indicators; test uncertainty and OOD detection; then assess repeatability across surveys before operational use.

## 14. Quick glossary

| Term | Plain-language meaning |
|---|---|
| Benthic | Related to the seafloor or bottom-dwelling community |
| Class map | Image-sized grid whose integer values identify the predicted category per pixel |
| Dense prediction | A prediction made at every pixel location |
| Domain shift | New imagery differs from the data a model learned from |
| Entropy | Measure of how spread out class probabilities are; higher is less decisive |
| Fine-tuning | Further training a pretrained model on the target task |
| Ground truth | Human/reference annotation used for training or evaluation |
| IoU | Overlap score between predicted and reference pixels for a class |
| LoRA | Parameter-efficient low-rank updates used to adapt a large model |
| mIoU | Mean IoU over classes; useful when class sizes differ |
| Mask | Binary or class-valued pixel map identifying image regions |
| Non-reef sink | Project category used to hold predicted non-benthic scene pixels |
| Semantic segmentation | Per-pixel categories without separating same-class objects into instances |
| Softmax | Converts class scores into probabilities summing to one |
| Test split | Held-out examples used for final measurement, not model tuning |
| Transformer | Neural-network architecture that uses attention to mix information across input tokens/patches |
| ViT | Vision Transformer; applies transformer blocks to image patches |

## 15. Project files to know for the presentation

- `main.py` — command-line entry point (`infer`, `evaluate`, `prepare`, `app`).
- `src/inference/pipeline.py` — image/video flow, health formulas, overlays, frame aggregation.
- `src/data/label_map.py`, `configs/labels.yaml`, `configs/label_map.json` — project taxonomy and mapping.
- `src/data/prepare.py` — Parquet extraction and YOLO label conversion.
- `src/data/quality.py` — blur, exposure, contrast, and color-cast heuristics.
- `src/trust/reliability.py` — confidence/entropy helpers and review decision.
- `src/evaluate/segmentation.py`, `scripts/evaluate_model_comparison.py` — metric calculation and benchmark runner.
- `src/inference/dinov3_dpt.py` — DINOv3 checkpoint loading and label mapping.
- `src/app/streamlit_app.py` — interactive dashboard.
- `kaggle/` — cloud training scripts.
- `model_comparison.md` and `models/model_comparison.json` — detailed scores and protocols.
- `docs/paper-inventory.md`, `docs/dataset-inventory.md`, `docs/btp-approach.md` — project research and design context.

## Research references for the literature section

- Bridge et al. (2024), [Improving coral monitoring by reducing variability and bias in cover estimates from seabed images](https://doi.org/10.1016/j.pocean.2024.103214) — observer variability and the need for standardized estimates.
- Sauder et al. (2025), [The Coralscapes Dataset: Semantic Scene Understanding in Coral Reefs](https://openaccess.thecvf.com/content/ICCV2025W/CVAUI%20%26%20AAMVEM/papers/Sauder_The_Coralscapes_Dataset_Semantic_Scene_Understanding_in_Coral_Reefs_ICCVW_2025_paper.pdf) — dense reef-scene segmentation dataset.
- Sauder et al. (2026), [CoralscapesV2 preprint](https://arxiv.org/abs/2609.12826) and [dataset card](https://huggingface.co/datasets/josauder/coralscapesV2) — 95-class extension used by this project.
- Xie et al. (2021), [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://proceedings.neurips.cc/paper_files/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html) — hierarchical transformer segmentation architecture.
- Siméoni et al. (2025), [DINOv3](https://arxiv.org/abs/2508.10104), [official implementation](https://github.com/facebookresearch/dinov3), and [DPT dense-prediction paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Ranftl_Vision_Transformers_for_Dense_Prediction_ICCV_2021_paper.pdf) — self-supervised visual representations and transformer dense prediction.
- [Ultralytics YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/) — official model family and task documentation.
- González-Rivero et al. (2020), Kopecky et al. (2023), Wyatt et al. (2025/2026), CoralVOS (2025), DeepReefMap, and other papers are catalogued in [`docs/paper-inventory.md`](paper-inventory.md). The inventory identifies each work's task and why its metrics are or are not comparable to this project.
