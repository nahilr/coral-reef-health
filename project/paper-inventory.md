# Paper Inventory

**Inventory checked:** 2026-10-07
**Project:** Trustworthy AI-Assisted Coral Reef Health Assessment

This file indexes the reference papers collected for the project. Local PDF copies live under `papers/` (active) and `papers/archive/` (not in the active pipeline); the PDFs are **not tracked in git** — the links below are the canonical web sources.

## Active references (`papers/`)

| Paper | Role in project | Link |
|---|---|---|
| **Sauder et al. 2025 — The Coralscapes Dataset: Semantic Scene Understanding in Coral Reefs** (ICCVW 2025) | Source of the entire dataset: 2,075 images, 39 benthic classes; basis of the 7-class label map | [CVF open access](https://openaccess.thecvf.com/content/ICCV2025W/CVAUI%20%26%20AAMVEM/papers/Sauder_The_Coralscapes_Dataset_Semantic_Scene_Understanding_in_Coral_Reefs_ICCVW_2025_paper.pdf) · [HF dataset](https://huggingface.co/datasets/EPFL-ECEO/coralscapes) |
| **González-Rivero et al. 2020 — Monitoring of Coral Reefs Using Artificial Intelligence: A Feasible and Cost-Effective Approach** (Remote Sensing 12(3):489) | Main methodological precedent for automated reef benthic monitoring | [doi:10.3390/rs12030489](https://doi.org/10.3390/rs12030489) |
| **Sauder et al. 2025 — Rapid consistent reef surveys with DeepReefMap** (Scientific Reports) | Frame sampling and survey-level aggregation ideas for video inference | [doi:10.1038/s41598-025-20795-z](https://doi.org/10.1038/s41598-025-20795-z) |
| **Wyatt et al. 2025 — Safe AI for coral reefs: Benchmarking out-of-distribution detection algorithms for coral reef image surveys** (Ecological Informatics 90:103207) | OOD-detection benchmark informing the trust/reliability gate | [doi:10.1016/j.ecoinf.2025.103207](https://doi.org/10.1016/j.ecoinf.2025.103207) |
| **Wyatt et al. 2026 — Beyond classification accuracy: Uncertainty-aware deep learning for coral reef monitoring** (Ecological Informatics 96:103831) | Uncertainty-aware monitoring; motivation for reporting uncertainty alongside predictions | [doi:10.1016/j.ecoinf.2026.103831](https://doi.org/10.1016/j.ecoinf.2026.103831) |
| **Dey et al. 2025 — Local Environmental Filtering and Frequency of Marine Heatwaves Influence Decadal Trends in Coral Composition** (Diversity and Distributions) | Lakshadweep ecological context and comparison targets | [doi:10.1111/ddi.70043](https://doi.org/10.1111/ddi.70043) |
| **Guo et al. 2017 — On Calibration of Modern Neural Networks** (ICML) | ECE and temperature scaling used by the trust layer | [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) |
| **Hendrycks & Gimpel 2017 — A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks** (ICLR) | Max-softmax baseline behind the confidence signals | [arXiv:1610.02136](https://arxiv.org/abs/1610.02136) |

## Archived (`papers/archive/`)

Collected but not part of the active pipeline.

| Paper | Link |
|---|---|
| Ouassine et al. 2025 — Deep learning for automated coral reef monitoring: a novel system based on YOLOv8 detection and DeepSORT tracking (Ecological Informatics 89:103170) | [doi:10.1016/j.ecoinf.2025.103170](https://doi.org/10.1016/j.ecoinf.2025.103170) |
| Gapper et al. 2024 — A generalized machine learning model for long-term coral reef monitoring in the Red Sea (Heliyon 10:e38249) | [doi:10.1016/j.heliyon.2024.e38249](https://doi.org/10.1016/j.heliyon.2024.e38249) |
| Shao et al. 2024 — Deep learning for multi-label classification of coral conditions in the Indo-Pacific via underwater photogrammetry | [arXiv:2403.05930](https://arxiv.org/abs/2403.05930) |
| Sauder et al. 2025 — The Coralscapes Dataset (arXiv preprint of the ICCVW paper above) | [arXiv:2503.20000](https://arxiv.org/abs/2503.20000) |
| Shao et al. 2025 — Multi-label classification for multi-temporal, multi-spatial coral reef condition monitoring using vision foundation model with adapter learning | [arXiv:2503.23012](https://arxiv.org/abs/2503.23012) |
| Malik et al. 2026 — Advancing coral reef monitoring: a deep learning perspective on automated segmentation and classification (Discover Applied Sciences 8:217) | [doi:10.1007/s42452-025-07831-3](https://doi.org/10.1007/s42452-025-07831-3) |
| Kopecky et al. 2023 — Quantifying the Loss of Coral from a Bleaching Event Using Underwater Photogrammetry and AI-Assisted Image Segmentation (Remote Sensing 15(16):4077) | [doi:10.3390/rs15164077](https://doi.org/10.3390/rs15164077) |
| A comprehensive literature review on coral reef health monitoring: techniques, status, and emerging trends with special focus on South India (Machine Learning for Computational Science and Engineering, 2025, 1:36) | [doi:10.1007/s44379-025-00036-w](https://doi.org/10.1007/s44379-025-00036-w) |
| Zheng et al. 2025 — CoralVOS: Dataset and Benchmark for Dense Coral Video Segmentation (ICCVW 2025) | [CVF open access](https://openaccess.thecvf.com/content/ICCV2025W/CVAUI%20%26%20AAMVEM/html/Zheng_CoralVOS_Dataset_and_Benchmark_for_Dense_Coral_Video_Segmentation_ICCVW_2025_paper.html) |
| Zheng et al. 2023 — CoralVOS preprint | [arXiv:2310.01946](https://arxiv.org/abs/2310.01946) |
| Chen et al. 2021 — A New Deep Learning Engine for CoralNet (ICCVW 2021) | [CVF open access](https://openaccess.thecvf.com/content/ICCV2021W/OceanVision/html/Chen_A_New_Deep_Learning_Engine_for_CoralNet_ICCVW_2021_paper.html) |
| Akkaynak & Treibitz 2019 — Sea-thru: A Method For Removing Water From Underwater Images (CVPR 2019) | [CVF open access](https://openaccess.thecvf.com/content_CVPR_2019/html/Akkaynak_Sea-Thru_A_Method_for_Removing_Water_From_Underwater_Images_CVPR_2019_paper.html) |

## Notes

- Analysis and synthesis of these sources: [literature-review.md](literature-review.md); verified dataset/technical links: [research-sources.md](research-sources.md).
- Cite the published versions (not the local PDFs) in any report or paper.
