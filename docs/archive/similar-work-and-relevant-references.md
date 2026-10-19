# Similar Work and Relevant References

Checked 2026-09-16.

## Papers most similar to the proposed system

| Paper | What it achieves | Data/source | Relevance |
|---|---|---|---|
| González-Rivero et al. (2020), *Monitoring of Coral Reefs Using Artificial Intelligence* | Automated benthic annotation and coral-cover estimation for monitoring and change detection | Global reef monitoring imagery; images and annotations are linked from the paper's data archive | Core ecological-AI precedent |
| Sauder et al. (2025), *Rapid consistent reef surveys with DeepReefMap* | Processes underwater transect video into semantic 3D maps and benthic-cover estimates | Red Sea videos from Djibouti, Jordan, and Israel; example videos and annotated frames are on Zenodo | Closest practical ROV/video workflow |
| Sauder et al. (2025), *The Coralscapes Dataset* | Dense semantic segmentation of general reef scenes | 2,075 images, 39 benthic classes, 174,077 masks | Best training/benchmark source for segmentation |
| Ouassine et al. (2025), *Deep learning for automated coral reef monitoring: YOLOv8 and DeepSORT* | Detects and tracks coral formations in underwater video | AIMECORAL1: 580 images from the Southwest Indian Ocean; AIMECORAL2: 282 images from New Caledonia | Useful for detection/tracking, but not sufficient alone for reef-health estimation |
| Shao et al. (2025), *Multi-label classification for multi-temporal, multi-spatial coral reef condition monitoring* | Uses DINOv2 + LoRA for multi-label coral-condition classification across sites and seasons | 42,105 patches from 1,203 underwater images at 15 Koh Tao dive sites | Useful for condition classification and spatial/temporal generalization |
| Wyatt et al. (2025), *Safe AI for coral reefs* | Tests OOD detection under changing reef-image distributions | Public coral image datasets and source/geography shifts; code/data are on GitHub and Zenodo | Core trustworthy-AI reference |
| Wyatt et al. (2026), *Beyond classification accuracy* | Propagates calibrated model uncertainty into coral-cover estimates | Simulated and real coral benthic datasets | Core reference for uncertainty-aware change estimates |
| CoralVOS (2023), *Dataset and Benchmark for Coral Video Segmentation* | Builds a coral video-object-segmentation benchmark | Video segmentation data described in the paper/repository | Relevant if the project requires object persistence through video |

## Source links

- González-Rivero et al.: https://www.coralreefecosystems.org/wp-content/uploads/2015/01/Gonzalez-Rivero-et-al.-2020-Monitoring-of-Coral-Reefs-Using-Artificial-Intelli.pdf
- DeepReefMap paper: https://www.nature.com/articles/s41598-025-20795-z
- DeepReefMap example videos and annotations: https://zenodo.org/records/10624794
- DeepReefMap code: https://github.com/eceo-epfl/deepreefmap
- Coralscapes paper: https://openaccess.thecvf.com/content/ICCV2025W/CVAUI%20%26%20AAMVEM/papers/Sauder_The_Coralscapes_Dataset_Semantic_Scene_Understanding_in_Coral_Reefs_ICCVW_2025_paper.pdf
- Coralscapes dataset/research page: https://reefnet-project.github.io/reefnet-2025/
- YOLOv8/DeepSORT paper: https://doi.org/10.1016/j.ecoinf.2025.103170
- DINOv2 + LoRA paper: https://arxiv.org/abs/2503.23012
- Safe AI paper: https://doi.org/10.1016/j.ecoinf.2025.103207
- Safe AI code/data: https://github.com/threehundred/safeai-for-coral-reefs-public
- Uncertainty-aware paper: https://doi.org/10.1016/j.ecoinf.2026.103831
- CoralVOS: https://arxiv.org/abs/2310.01946

## References from these papers that are directly relevant

### 1. Beijbom et al. (2012) — Automated Annotation of Coral Reef Survey Images

Use for the basic point-based benthic classification formulation and the Moorea LTER dataset. It supports the idea that random point annotations can be converted into coral-cover estimates.

### 2. Chen et al. (2021) — A New Deep Learning Engine for CoralNet

Use for the CoralNet/pyspacer annotation workflow and as background for automated point classification. This is relevant if the project uses CoralNet or MERMAID-style labels.

### 3. Akkaynak and Treibitz (2019) — Sea-Thru

Use for underwater image formation, colour distortion, and image enhancement. It is relevant to the image-quality stage, but enhancement should be evaluated carefully because it may alter class evidence.

### 4. Hendrycks and Gimpel (2017) — A Baseline for Detecting Misclassified and Out-of-Distribution Examples

Use as the general OOD baseline behind confidence-based and score-based “I do not know” mechanisms.

### 5. Liang et al. (2018) — Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks

Use for ODIN-style OOD detection as a comparison method. Do not assume it will be the best method for reef imagery; the Safe AI coral study found OOD performance to be dataset-dependent.

### 6. Gal and Ghahramani (2016) — Dropout as a Bayesian Approximation

Use as a theoretical basis for Monte Carlo dropout uncertainty. Compare it with calibration and simpler uncertainty estimates rather than treating it as automatically reliable.

### 7. Guo et al. (2017) — On Calibration of Modern Neural Networks

Use for temperature scaling and calibration evaluation. This is directly relevant to making model confidence meaningful.

### 8. Akkaynak et al. / underwater imaging references cited by the review papers

Use to justify why turbidity, colour attenuation, lighting, blur, and camera variation create distribution shift. This supports your image-quality and OOD experiments.

## Recommended citation hierarchy for your BTP

### Must cite and discuss in depth

1. Lakshadweep long-term ecological study by Dey et al.
2. González-Rivero et al. (2020)
3. DeepReefMap (2025)
4. Safe AI for coral reefs (2025)
5. Uncertainty-aware deep learning (2026)

### Cite in the methodology section

6. Coralscapes
7. Beijbom et al. (2012)
8. Guo et al. (2017)
9. Hendrycks and Gimpel (2017)
10. Akkaynak and Treibitz (2019)

### Cite only if those components are implemented

11. YOLOv8/DeepSORT — if object detection and tracking are used.
12. DINOv2 + LoRA — if parameter-efficient foundation-model adaptation is tested.
13. CoralVOS — if video-object segmentation or persistent coral-object masks are used.

## Main research gap for this BTP

Existing work commonly solves one of these problems:

- benthic classification,
- coral-cover estimation,
- video/3D mapping,
- bleaching classification,
- OOD detection, or
- uncertainty estimation.

Your defensible contribution is to integrate these into a smaller, auditable workflow that reports reef-condition indicators together with calibration, OOD status, and uncertainty. The contribution should be the integration and evaluation protocol, not the invention of every individual model.

