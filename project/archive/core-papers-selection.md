# Core Papers for the BTP

## Project title

Trustworthy AI-Assisted Coral-Reef Health Assessment and Temporal Change Detection in Lakshadweep

## Recommended main-paper set

### 1. Dey et al. — Lakshadweep long-term coral responses

**Paper:** *Local Environmental Filtering and Frequency of Marine Heatwaves Influence Decadal Trends in Coral Composition* (Diversity and Distributions, 2025)

**DOI:** https://doi.org/10.1111/ddi.70043

**Role in the BTP:** This should be the ecological and regional foundation. It establishes the Lakshadweep monitoring context, repeated surveys, coral composition, bleaching events, depth, wave exposure, and long-term recovery patterns.

**What to borrow:**

- Site/depth/exposure as meaningful analysis units.
- Longitudinal coral-cover and composition analysis.
- Careful interpretation of bleaching and recovery rather than a simplistic health score.

**Why it is essential:** Without this paper, the AI work can become disconnected from the actual Lakshadweep monitoring problem.

### 2. González-Rivero et al. — AI for ecological monitoring and change detection

**Paper:** *Monitoring of Coral Reefs Using Artificial Intelligence: A Feasible and Cost-Effective Approach* (Remote Sensing, 2020)

**Local file:** `Monitoring_of_Coral_Reefs_Using_Artificial_Intelli.pdf`

**Role in the BTP:** This should be the main methodological baseline for automated benthic annotation and temporal monitoring.

**What to borrow:**

- Convert image-level predictions into benthic-composition or coral-cover estimates.
- Evaluate agreement with expert observations.
- Aggregate observations to transect/site level before comparing time points.
- Evaluate whether automated analysis preserves the ability to detect ecological change.

**Why it is essential:** It connects computer vision to an ecological monitoring outcome instead of treating classification accuracy as the final result.

### 3. Sauder et al. — Semantic mapping and quantitative video surveying

**Paper:** *Rapid consistent reef surveys with DeepReefMap* (Scientific Reports, 2025)

**Local file:** `s41598-025-20795-z.pdf`

**Role in the BTP:** This should guide the ROV/video and semantic-segmentation direction if local video becomes available.

**What to borrow:**

- Video transects as the sampling unit.
- Semantic segmentation for benthic cover rather than only object boxes.
- Quality and consistency checks across video conditions.
- Separation of data acquisition, mapping, segmentation, and ecological reporting.

**Why it is essential:** It is closer to the proposed ROV monitoring workflow than ordinary image classification papers.

### 4. Wyatt et al. — Safe AI and out-of-distribution detection

**Paper:** *Safe AI for coral reefs: Benchmarking out-of-distribution detection algorithms for coral reef image surveys* (Ecological Informatics, 2025)

**Local file:** `1-s2.0-S157495412500216X-main.pdf`

**Role in the BTP:** This should be the principal trustworthy-AI paper.

**What to borrow:**

- Treat geography, sensor, lighting, turbidity, depth, and survey conditions as distribution shifts.
- Compare confidence with an independent OOD signal.
- Use held-out datasets or sources to test whether the system knows when it is unfamiliar.
- Add a human-review or abstention path.

**Why it is essential:** It gives the “trustworthy AI” part a concrete evaluation design rather than making trust a vague dashboard feature.

### 5. Wyatt et al. — Uncertainty propagation into coral-cover estimates

**Paper:** *Beyond classification accuracy: Uncertainty-aware deep learning for coral reef monitoring* (Ecological Informatics, 2026)

**Local file:** `1-s2.0-S1574954126002372-main.pdf`

**Role in the BTP:** This should guide uncertainty estimation for aggregate ecological measurements.

**What to borrow:**

- Calibrated classifiers instead of raw softmax confidence.
- Monte Carlo sampling or another defensible method for propagating prediction uncertainty.
- Uncertainty intervals for estimated coral cover and change.
- Separate classification error from uncertainty in the final ecological metric.

**Why it is essential:** Your project reports temporal change, so uncertainty must apply to the change estimate—not only to individual image predictions.

## Best five-paper narrative

```text
Lakshadweep ecological baseline
              ↓
Automated benthic monitoring
              ↓
Semantic/video mapping
              ↓
OOD-aware trustworthy predictions
              ↓
Uncertainty-aware temporal change estimates
```

## Supporting papers, not main foundations

### Ouassine et al. — YOLOv8 and DeepSORT

Useful for object detection and video tracking, but it should remain a supporting implementation paper. Tracking objects through one continuous video does not prove that the same coral colony is being observed across different survey years. Use it only if the local data are genuinely sequential ROV video and colony identity is required.

### Shao et al. — DINOv2 with LoRA

Useful as a transfer-learning experiment for multi-label coral-condition classification. It should not define the whole project because the current Lakshadweep data do not provide image-level condition labels, and the paper's Koh Tao domain is different.

### Sauder et al. — Coralscapes Dataset

Useful as a dataset/segmentation reference and a possible external training source. It complements DeepReefMap but is not itself the ecological basis for the Lakshadweep case study.

### Gapper et al. — Generalized machine learning model in the Red Sea

Useful for background on spatial and temporal generalization in remote sensing. It is less central if the BTP focuses on underwater imagery and Lakshadweep photoquadrats.

### Malik et al. — Deep-learning review

Useful for the literature-review chapter and model taxonomy, but not as a primary methodological foundation. The project should derive its experimental design from the five papers above and the actual available datasets.

## Suggested literature-review structure

1. Lakshadweep coral ecology and monitoring problem — Dey et al.
2. Automated image analysis for reef monitoring — González-Rivero et al.
3. Quantitative semantic mapping from video — DeepReefMap.
4. Reliability under distribution shift — Wyatt et al. Safe AI.
5. Uncertainty in ecological estimates — Wyatt et al. uncertainty-aware monitoring.
6. Supporting model choices — YOLOv8/DeepSORT, DINOv2/LoRA, Coralscapes, and related reviews.

## One-sentence justification for the mentor

> “We selected the Lakshadweep long-term monitoring work as our ecological foundation, automated benthic monitoring and DeepReefMap as our computer-vision foundations, and the two Wyatt papers as our trustworthy-AI foundations for out-of-distribution detection and uncertainty-aware temporal estimates.”

