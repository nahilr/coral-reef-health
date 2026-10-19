# Base papers and proposed contribution

## BTP scope

**Trustworthy AI-Assisted Coral-Reef Health Assessment and Temporal Change Detection in Lakshadweep**

The proposed system should take georeferenced ROV video, produce auditable benthic/coral measurements, estimate condition-related indicators, compare surveys across time, and report when its predictions are uncertain or outside its training experience.

The papers below are grouped into: (1) the main foundation papers, (2) supporting model/dataset papers, and (3) trust and preprocessing papers.

## A. Main foundation papers

### 1. Dey et al. (2025) — Lakshadweep coral composition and marine heatwaves

**Paper:** *Local Environmental Filtering and Frequency of Marine Heatwaves Influence Decadal Trends in Coral Composition*

**What it does:** Uses long-term ecological observations to study how coral composition changes across Lakshadweep locations and how local environmental conditions, depth, exposure, and marine heatwave history influence those trends.

**Speciality/uniqueness:** It is the regional ecological anchor for this BTP. It prevents the computer-vision project from inventing an arbitrary “health score” disconnected from the actual reef processes and monitoring units of Lakshadweep.

**Gap for our problem:** It is not an ROV-video AI system. It does not provide a complete automated pipeline from underwater video to calibrated health indicators, nor does it solve frame-level domain shift, uncertainty, or human review.

**What we can add:** Use its ecological variables and site/depth/exposure structure to define sampling units and interpret change. Link AI-derived coral cover, benthic composition, bleaching/disease indicators, and image-quality metadata to the environmental context. The AI output should support ecological inference, not replace it.

### 2. González-Rivero et al. (2020) — AI-assisted coral-reef monitoring

**Paper:** *Monitoring of Coral Reefs Using Artificial Intelligence: A Feasible and Cost-Effective Approach*

**What it does:** Automates benthic annotation and converts image-level or point-level predictions into coral-cover and community-composition measurements for monitoring.

**Speciality/uniqueness:** It establishes the key bridge from computer vision to an ecological monitoring result: the final quantity is coral/benthic composition at a transect or site level, not just image classification accuracy.

**Gap for our problem:** It is primarily an image-annotation and monitoring workflow, not a Lakshadweep ROV-video system. It does not fully address temporal registration of video surveys, modern dense video segmentation, OOD detection, or uncertainty intervals for change estimates.

**What we can add:** Preserve its ecological aggregation idea, but implement it for ROV clips. Aggregate frame predictions into transect-level estimates, quantify effective sampling coverage, and report confidence intervals for coral-cover change between surveys.

### 3. Sauder et al. (2025) — DeepReefMap

**Paper:** *Rapid Consistent Reef Surveys with DeepReefMap*

**What it does:** Processes underwater survey video into spatially organized reef representations and uses semantic understanding to support quantitative benthic mapping and cover estimates.

**Speciality/uniqueness:** It is the closest practical precedent for a video-first workflow. Its emphasis is on repeatable survey mapping and consistency rather than isolated image classification.

**Gap for our problem:** It is not designed specifically for Lakshadweep, and its mapping output is not by itself a trustworthy health-assessment system. Cross-survey registration, biological condition labels, calibrated uncertainty, and explicit abstention remain open parts for our use case.

**What we can add:** Adapt the survey-unit and semantic-mapping ideas to Lakshadweep ROV transects. Add repeat-survey alignment, quality filtering, OOD flags, and uncertainty-aware reporting so that the system says both “what changed” and “how reliable that conclusion is.”

### 4. Wyatt et al. (2025) — Safe AI for coral reefs

**Paper:** *Safe AI for Coral Reefs: Benchmarking Out-of-Distribution Detection Algorithms for Coral Reef Image Surveys*

**What it does:** Evaluates whether models can detect inputs that differ from the training distribution, including changes in reef source, geography, imaging conditions, or other survey characteristics.

**Speciality/uniqueness:** It makes trustworthy AI operational for coral monitoring. A model is not treated as reliable merely because its average classification accuracy is high; it must recognize unfamiliar inputs.

**Gap for our problem:** OOD detection does not itself produce ecological health estimates, temporal change, or a complete video pipeline. Its benchmark conditions may not represent Lakshadweep ROV footage, and OOD scores still require threshold selection and field validation.

**What we can add:** Build a Lakshadweep-specific shift benchmark: train on general-reef data, test on held-out reef/geography/camera/turbidity conditions, and measure whether OOD alerts correspond to genuine expert disagreement or poor image quality. Route flagged clips to human review or exclude them from automated change estimates.

### 5. Wyatt et al. (2026) — Uncertainty-aware coral monitoring

**Paper:** *Beyond Classification Accuracy: Uncertainty-Aware Deep Learning for Coral Reef Monitoring*

**What it does:** Studies calibrated predictive uncertainty and propagates model uncertainty into ecological quantities such as coral-cover estimates.

**Speciality/uniqueness:** It addresses the level that matters for conservation decisions: uncertainty in the final ecological metric, not only uncertainty for one image or one class.

**Gap for our problem:** The methods need to be adapted to sequential ROV video, spatial correlation between frames, site-level sampling, and Lakshadweep-specific labels. Uncertainty alone also does not reveal whether a sample is OOD; those are related but distinct signals.

**What we can add:** Combine calibration, frame/clip aggregation, bootstrap or hierarchical sampling, and OOD status to produce an interval for temporal change. Report separate sources of uncertainty: model uncertainty, sampling uncertainty, image-quality uncertainty, and domain-shift risk.

## B. Supporting model and dataset papers

### 6. Sauder et al. (2025) — Coralscapes Dataset

**What it does:** Provides a large semantic-segmentation resource for coral-reef scenes with many benthic categories and dense masks.

**Speciality/uniqueness:** It gives a practical pretraining and external-benchmark source for general reef scene understanding.

**Gap:** It is not Lakshadweep-specific and its labels do not automatically encode reef health, bleaching severity, disease, or temporal identity.

**What we can add:** Pretrain or benchmark segmentation on Coralscapes, then fine-tune with a smaller Lakshadweep dataset. Measure the performance drop from general reefs to Lakshadweep and use that drop as evidence for domain adaptation and OOD evaluation.

### 7. CoralVOS (2023/2025) — Dense coral video segmentation

**What it does:** Provides a benchmark for segmenting coral objects across video frames, including the temporal aspect of underwater imagery.

**Speciality/uniqueness:** It is more directly video-oriented than ordinary reef-image datasets and supports investigation of temporal consistency.

**Gap:** Video object segmentation is not the same as ecological health assessment. It does not by itself define coral-cover indicators, health labels, or Lakshadweep transfer performance.

**What we can add:** Use temporal consistency as a reliability signal. If predictions flicker strongly across adjacent frames, reduce confidence, aggregate at clip level, or request review. Extend this to repeat-survey change only after handling spatial registration and sampling differences.

### 8. Ouassine et al. (2025) — YOLOv8 and DeepSORT

**Paper:** *Deep Learning for Automated Coral Reef Monitoring: A Novel System Based on YOLOv8 Detection and DeepSORT Tracking*

**What it does:** Detects coral formations and tracks them across consecutive video frames.

**Speciality/uniqueness:** It provides a straightforward object-detection and within-video tracking baseline.

**Gap:** Detection/tracking boxes are not equivalent to benthic cover or reef health. Tracking the same object within one clip does not establish that it is the same coral colony across different years.

**What we can add:** Use tracking only where it improves clip-level stability or colony-level analysis. Compare it against semantic segmentation and explicitly avoid claiming cross-year colony identity without georeferencing, repeatable camera trajectories, or a validated registration method.

### 9. Shao et al. (2025) — DINOv2 with LoRA

**Paper:** *Multi-label Classification for Multi-temporal, Multi-spatial Coral Reef Condition Monitoring*

**What it does:** Adapts a visual foundation model using parameter-efficient fine-tuning for multi-label coral-condition classification across sites and time periods.

**Speciality/uniqueness:** It is a useful modern transfer-learning strategy when labelled coral-condition data are limited.

**Gap:** Its labels and environment are not Lakshadweep ROV labels. Multi-label condition prediction can be confounded by lighting, turbidity, viewpoint, and label subjectivity.

**What we can add:** Test DINOv2/LoRA as one classification branch, but require calibration, site-held-out testing, and label-agreement analysis. Compare it with segmentation-derived health indicators rather than assuming a whole-image condition label is sufficient.

### 10. Gapper et al. (2024) — Generalized machine learning model for Red Sea monitoring

**What it does:** Studies a machine-learning model intended to generalize across longer-term and/or spatially varying coral-reef monitoring observations in the Red Sea.

**Speciality/uniqueness:** It is relevant for thinking about spatial and temporal generalization instead of random train/test splits.

**Gap:** The geographic, ecological, and sensing context differs from Lakshadweep, and it is not a complete ROV semantic-video and trustworthy-AI pipeline.

**What we can add:** Borrow its generalization mindset: test by site, date, camera, and environmental condition. Avoid frame-level random splits that leak nearly identical neighboring frames into train and test sets.

### 11. Chen et al. (2021) — Deep-learning engine for CoralNet

**What it does:** Describes a deep-learning engine and annotation workflow for automated point-based coral/benthic classification in CoralNet.

**Speciality/uniqueness:** It supports scalable expert annotation and point-based cover estimation, which is practical when dense pixel masks are too expensive.

**Gap:** Point labels do not capture every object boundary or temporal relationship, and the workflow does not solve ROV video consistency or trustworthy change detection.

**What we can add:** Use sparse point labels as a low-cost first annotation layer, then reserve dense masks for a smaller validation set. This creates a realistic BTP annotation strategy and lets us compare point-based cover estimates with dense segmentation estimates.

### 12. Beijbom et al. (2012) — Automated annotation of coral-reef survey images

**What it does:** Establishes an early automated image-annotation formulation for coral-reef survey images and point-based benthic classification.

**Speciality/uniqueness:** It is a foundational reference for converting sampled image points into benthic-cover measurements and for using expert-labelled reef imagery as a supervised-learning problem.

**Gap:** It predates current video models, foundation models, OOD detection, calibration, and modern underwater image-quality handling. It also does not address Lakshadweep or temporal change reliability.

**What we can add:** Use it to justify the sampling and annotation formulation, then update the pipeline with video clips, transfer learning, explicit domain-shift testing, and uncertainty-aware ecological reporting.

## C. Trust, preprocessing, and general methodology

## D. Newly added papers

### 19. Pandiyarajan et al. (2025) — Coral-reef health monitoring review with a South India focus

**Paper:** *A Comprehensive Literature Review on Coral Reef Health Monitoring: Techniques, Status, and Emerging Trends with Special Focus on South India*

**What it does:** Reviews field surveys, remote sensing, machine learning, CoralNet, satellite bleaching products, underwater imaging, photogrammetry, microbial indicators and emerging AI-assisted monitoring, with specific discussion of Lakshadweep, Gulf of Mannar and Palk Bay.

**Speciality/uniqueness:** It is the most regionally relevant review in the collection. It helps connect the BTP to South Indian monitoring realities, cost and feasibility rather than presenting the project as a generic global computer-vision problem.

**Gap:** It is a review, not a new dataset or validated ROV-health model. Some technologies discussed, such as microbial diagnostics, hyperspectral imaging and large-scale UAV mapping, are outside the six-month BTP scope.

**What we can add:** Use it to justify the selected monitoring indicators, explain why a low-cost RGB/ROV workflow is practical, and position the BTP as one computational component of a broader monitoring system. Do not attempt to implement every method reviewed.

### 20. Kopecky et al. (2023) — Photogrammetry and AI-assisted bleaching-event change detection

**Paper:** *Quantifying the Loss of Coral from a Bleaching Event Using Underwater Photogrammetry and AI-Assisted Image Segmentation*

**What it does:** Combines underwater photogrammetry and AI-assisted segmentation to compare live and dead coral surface area before and after a bleaching event. It shows that approximate 3D surface area can reveal changes that a planar 2D measurement may underestimate.

**Speciality/uniqueness:** It is the closest new paper to the temporal-change and “health impact” part of the BTP. It links image segmentation to an ecological disturbance outcome rather than stopping at classification accuracy.

**Gap:** It requires carefully designed repeat photogrammetry, 3D reconstruction and matched reef surveys. That is too ambitious as the core of a six-month BTP unless suitable repeat data already exist.

**What we can add:** Borrow the comparison logic—live/dead composition and before/after change—but implement a simpler 2D image-derived composition estimate. Mention 3D photogrammetry as future work unless local repeat imagery and camera geometry are available.

### 21. Shao et al. (2024/2025) — Multi-label coral-condition classification from underwater photogrammetry

**Paper:** *Deep Learning for Multi-Label Classification of Coral Conditions in the Indo-Pacific via Underwater Photogrammetry*

**What it does:** Builds a dataset of more than 20,000 high-resolution coral images and classifies conditions such as healthy, compromised, dead and rubble, together with stressors including competition, disease, predation and physical damage. It evaluates several deep-learning models and proposes an ensemble approach.

**Speciality/uniqueness:** It is the most directly relevant paper for a health-classification version of the simplified BTP. It shows that coral condition can be treated as a multi-label problem rather than only live/dead segmentation.

**Gap:** The data come from the Indo-Pacific but are not Lakshadweep-specific. Stressor labels can be subjective and the paper itself identifies generalization as an open problem. Multi-label condition classification also needs careful label definitions and expert validation.

**What we can add:** Use its label concepts to define an optional secondary task, but keep the BTP core to broad condition classes. Test transfer to a held-out source and add confidence/quality flags. Do not attempt the full stressor taxonomy unless matching local labels are obtained.

### 13. Akkaynak & Treibitz (2019) — Sea-Thru

**What it does:** Models underwater image formation and proposes a method for removing or reducing water-related colour and contrast effects.

**Speciality/uniqueness:** It treats underwater appearance as a physical image-formation problem rather than only applying generic enhancement.

**Gap:** Restoration can introduce artifacts or remove visual evidence useful to a classifier. It is not a health-assessment or temporal-change method.

**What we can add:** Treat restoration as an ablation: compare raw, colour-corrected, and quality-filtered inputs. Keep the version that improves cross-condition reliability, not merely visual appearance or one test-set score.

### 14. Guo et al. (2017) — Calibration of modern neural networks

**What it does:** Shows that neural-network confidence is often poorly calibrated and studies temperature scaling and related calibration methods.

**Speciality/uniqueness:** It provides the basis for making a reported probability more meaningful as a probability of correctness.

**Gap:** It is a general classification paper and does not address reef-video correlation, OOD conditions, or uncertainty in aggregate coral-cover change.

**What we can add:** Calibrate on a validation set separated by site or survey, report reliability diagrams and expected calibration error, and propagate calibrated predictions into clip/site-level metrics.

### 15. Hendrycks & Gimpel (2017) — OOD and misclassification baseline

**What it does:** Provides simple confidence-based baselines for detecting likely errors and out-of-distribution examples.

**Speciality/uniqueness:** It gives a transparent baseline against which more sophisticated OOD methods can be compared.

**Gap:** Maximum softmax confidence can fail under severe shift and is not a complete safety mechanism.

**What we can add:** Include it as the simplest baseline, then compare against energy/feature-based or ensemble methods. Evaluate not only AUROC but also operational metrics: coverage versus accuracy, false alarms, missed unfamiliar clips, and the fraction routed to human review.

### 16. Liang et al. (2018) — ODIN

**What it does:** Proposes an OOD-detection method that modifies confidence scoring using temperature scaling and input perturbation.

**Speciality/uniqueness:** It is a widely used comparison method for OOD detection.

**Gap:** It may be sensitive to hyperparameters and image characteristics; success on standard image benchmarks does not guarantee success on underwater imagery.

**What we can add:** Use it only as a benchmark, tune it without access to the final test set, and compare it with simpler baselines and feature-space methods under real reef shifts.

### 17. Gal & Ghahramani (2016) — Monte Carlo dropout

**What it does:** Provides a Bayesian interpretation of dropout and motivates using repeated stochastic forward passes to estimate predictive uncertainty.

**Speciality/uniqueness:** It offers a relatively accessible uncertainty method that can be added without training a fully Bayesian neural network.

**Gap:** The resulting uncertainty is not automatically calibrated and may not reflect dataset shift or sampling uncertainty.

**What we can add:** Compare MC dropout with calibration, ensembles, and bootstrap aggregation. Do not call the output “true confidence” unless it is empirically validated against errors and expert disagreement.

### 18. Malik et al. — Deep-learning review for coral-reef monitoring

**What it does:** Surveys how deep learning is being used in coral-reef monitoring and summarizes common tasks and architectures.

**Speciality/uniqueness:** Useful for organizing the literature and motivating the need for automated monitoring.

**Gap:** A review is not a dataset, benchmark, or experimental foundation for the BTP.

**What we can add:** Use it for the literature-review chapter only. Base the actual experimental design on the primary ecological, video, segmentation, OOD, and uncertainty papers above.

## The research gap in one sentence

Existing papers usually solve one component—benthic annotation, semantic video mapping, condition classification, OOD detection, or uncertainty estimation—but they do not provide a small, auditable Lakshadweep ROV workflow that connects all of these components to a defensible temporal change estimate.

## What the BTP can do on top of the literature

### Proposed contribution

Build and evaluate a **trustworthy, uncertainty-aware ROV-video monitoring pipeline for Lakshadweep**, using general-reef datasets for pretraining and local Lakshadweep data for calibration and ecological validation.

### Concrete additions

1. **Video-to-ecology conversion:** convert frame/clip predictions into coral cover, live coral versus abiotic substrate, benthic composition, and selected visible stress indicators at transect/site level.
2. **Cross-domain evaluation:** train on general-reef data and test separately by Lakshadweep site, date, camera, depth, turbidity, and lighting condition.
3. **Temporal consistency:** aggregate adjacent frames, detect prediction flicker, and use video-quality and consistency checks before reporting a change.
4. **Trust layer:** combine calibrated confidence, OOD score, image-quality score, and temporal consistency into a transparent review status: accepted, review required, or rejected.
5. **Uncertainty-aware change:** report an estimate and interval for change, for example coral-cover change between surveys, rather than a single unsupported percentage.
6. **Human-in-the-loop validation:** allow experts to review low-confidence or OOD clips and measure how much review improves reliability.
7. **A realistic annotation protocol:** use sparse point labels for scale, dense masks for a smaller validation subset, and explicit label-agreement measurements.
8. **Ecological interpretation:** relate AI-derived indicators to Lakshadweep site/depth/exposure and marine-heatwave context, while clearly separating correlation from causal claims.

## Recommended final system architecture

```text
ROV video + metadata
        ↓
Frame quality and underwater-condition assessment
        ↓
Semantic segmentation / point classification
        ↓
Clip aggregation and temporal consistency checks
        ↓
Coral-cover and condition indicators
        ↓
Calibration + OOD detection + uncertainty propagation
        ↓
Site-level temporal change with review flags
        ↓
Ecological interpretation for Lakshadweep
```

## Minimum defensible BTP novelty

The BTP does not need to invent a new neural-network architecture. A strong and feasible novelty is the **integration and evaluation protocol**: showing whether a model trained with general reef data can produce calibrated, uncertainty-aware, OOD-screened coral-health and temporal-change estimates for Lakshadweep ROV video.
