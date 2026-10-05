# Literature review

## Project context

**Fixed project title:** *Trustworthy AI-Assisted Coral-Reef Health Assessment and Temporal Change Detection in Lakshadweep*

The current six-month implementation focuses on the reef-health assessment part of the title. It will estimate broad visible reef-condition indicators from underwater images or sampled ROV frames and attach basic confidence and image-quality warnings. Detailed temporal change detection remains a future extension unless compatible repeated imagery becomes available.

## 1. Introduction

Coral reefs are important marine ecosystems that support biodiversity, fisheries, coastal protection and local livelihoods. They are threatened by marine heatwaves, bleaching, pollution, sedimentation, disease, overfishing and physical disturbance. Monitoring reef condition is therefore necessary, but conventional surveys are labour-intensive and often depend on expert divers manually analysing large collections of photographs or video.

Computer vision offers a way to scale this work. Models can classify sampled image points, patches or pixels into benthic categories and then convert the predictions into ecological quantities such as live coral cover and substrate composition. Underwater video also offers a potentially efficient survey source. However, reef imagery is difficult: colour and contrast change with depth and water quality, neighbouring video frames are highly correlated, and a model trained in one reef region may fail in another.

For this reason, the proposed BTP treats reef-health assessment as both a prediction problem and a reliability problem. The model should estimate broad visible condition, but it should also identify low-quality or unfamiliar imagery rather than producing an unjustified confident answer.

## 2. Operational definition of reef health

In this project, reef health is not treated as a single universal score. It is represented by measurable visual indicators:

- live or healthy-looking coral;
- dead coral or rubble;
- algae or biological overgrowth;
- sand, rock or other substrate;
- optional bleaching or visible-stress fraction.

The core model will use four broad classes. The predicted proportions of these classes provide an image-derived reef-condition summary. This is a practical proxy for visible condition, not a complete measurement of biodiversity, disease, resilience, water quality or ecosystem function.

## 3. Comparative review of collected papers

The table includes every paper currently present in the BTP papers collection. The two CoralVOS files are different versions of the same research work, and the two Shao papers are related but separate condition-classification studies.

| Paper and authors | Main task/data | Main contribution or speciality | Limitation/gap | Relevance to this BTP |
|---|---|---|---|---|
| **A Comprehensive Literature Review on Coral Reef Health Monitoring: Techniques, Status, and Emerging Trends with Special Focus on South India** — R. Pandiyarajan, K. Manohar and Ameed Nazhurudeen (2025) | Reviews field surveys, remote sensing, ML, CoralNet, ROVs, UAVs, hyperspectral and microbial indicators, with South India focus | Strong regional framing for Lakshadweep and practical comparison of monitoring methods | Review only; does not provide a new model or dataset | Use in introduction, motivation, regional context and justification of a low-cost AI workflow |
| **Local Environmental Filtering and Frequency of Marine Heatwaves Influence Decadal Trends in Coral Composition** — Mayukh Dey, Teresa Alcoverro, Carmen Gómez, Nachiket Kelkar, Jordi F. Pagès, Wenzel Pinto, Shreya Yadav, Rucha Karkarey, Mayuresh Gangal, M. K. Ibrahim, Aaron Lobo, Elrika D'Souza, Vardhan Patankar and colleagues (2025) | Long-term Lakshadweep coral-composition observations from 1998–2022, including depth and wave-exposure context | Provides the ecological and regional foundation for the BTP | Not a computer-vision, ROV or automated-health system | Use to define the Lakshadweep problem and interpret model outputs without arbitrary health claims |
| **Monitoring of Coral Reefs Using Artificial Intelligence: A Feasible and Cost-Effective Approach** — Manuel González-Rivero, Oscar Beijbom, Alberto Rodriguez-Ramirez, Dominic E. P. Bryant, Anjani Ganase, Yeray Gonzalez-Marrero, Ana Herrera-Reveles, Emma V. Kennedy, Catherine J. S. Kim, Sebastian Lopez-Marcano, Kathryn Markey, Benjamin P. Neal, Kate Osborne, Catalina Reyes-Nivia, Eugenia M. Sampayo, Kristin Stolberg, Abbie Taylor, Julie Vercelloni, Mathew Wyatt and Ove Hoegh-Guldberg (2020) | Automated benthic annotation and conversion to coral-cover/community-composition measures | Directly links image prediction to ecological monitoring outputs | Limited treatment of modern video, domain shift and trustworthy prediction | Main methodological precedent for estimating reef-condition indicators rather than reporting only accuracy |
| **Rapid Consistent Reef Surveys with DeepReefMap** — Jonathan Sauder, Guilhem Banc-Prandi, Gabriela Perna, Ibrahim Souleiman Abdallah, Osama S. Saad, Mustafa Altaib Mohammed, Ali Al-Sawalmih, Anders Meibom and Devis Tuia (2025) | Underwater video, 3D semantic mapping and benthic segmentation across Red Sea sites | Closest collected paper to practical ROV/video survey processing | Full pipeline requires video mapping and 3D reconstruction; not Lakshadweep validation | Borrow frame sampling, aggregation and survey consistency ideas; keep 3D mapping as future work |
| **The Coralscapes Dataset: Semantic Scene Understanding in Coral Reefs** — Jonathan Sauder, Viktor Domazetoski, Guilhem Banc-Prandi, Gabriela Perna, Anders Meibom and Devis Tuia (2025) | 2,075 reef images, 39 benthic classes and dense segmentation masks | Strong general-reef segmentation benchmark with spatial site splits | Red Sea data; no Lakshadweep-specific health labels | Use for optional segmentation, pretraining or external testing |
| **Deep Learning for Multi-Label Classification of Coral Conditions in the Indo-Pacific via Underwater Photogrammetry** — Xinlei Shao, Hongruixuan Chen, Kirsty Magson, Jiaqi Wang, Jian Song, Jundong Chen and Jun Sasaki (2024/2025) | More than 20,000 coral images with healthy, compromised, dead, rubble and stressor labels | Most direct health-condition classification reference in the collection | Labels are not Lakshadweep-specific; stressor categories may be subjective and difficult to harmonize | Use to design optional condition labels; retain only broad labels for the BTP |
| **Multi-label Classification for Multi-temporal, Multi-spatial Coral Reef Condition Monitoring Using Vision Foundation Model with Adapter Learning** — Xinlei Shao, Hongruixuan Chen, Fan Zhao, Kirsty Magson, Jundong Chen, Peiran Li, Jiaqi Wang and Jun Sasaki (2025) | Multi-temporal/multi-spatial coral-condition classification using a vision foundation model and adapter learning | Relevant modern transfer-learning strategy for limited labelled data | More complex than necessary for the BTP; generalization to Lakshadweep remains open | Cite as an advanced alternative; use only if the main model is completed early |
| **Quantifying the Loss of Coral from a Bleaching Event Using Underwater Photogrammetry and AI-Assisted Image Segmentation** — Kai L. Kopecky, Gaia Pavoni, Erica Nocerino, Andrew J. Brooks, Massimiliano Corsini, Fabio Menna, Jordan P. Gallagher, Alessandro Capra, Cristina Castagnetti, Paolo Rossi, Armin Gruen, Fabian Neyer, Alessandro Muntoni, Federico Ponchio, Paolo Cignoni, Matthias Troyer, Sally J. Holbrook and Russell J. Schmitt (2023) | Uses underwater photogrammetry and AI segmentation to measure live/dead coral change before and after bleaching | Strong example of connecting AI measurements to disturbance impacts and ecological change | Requires matched photogrammetry and 3D reconstruction; too ambitious as the BTP core | Use to motivate future temporal analysis; implement only simple composition comparison if data allow |
| **Safe AI for Coral Reefs: Benchmarking Out-of-Distribution Detection Algorithms for Coral Reef Image Surveys** — Mathew Wyatt, Sharyn Hickey, Ben Radford, Manuel Gonzalez-Rivero, Nader Boutros, Nikolaus Callow, Nicole Ryan, Arjun Chennu, Mohammed Bennamoun and James Gilmour (2025) | Tests OOD detection under changes in reef-image distribution | Makes trustworthy AI concrete by evaluating whether models recognise unfamiliar inputs | OOD performance depends on the chosen shifts and thresholds; does not itself estimate reef health | Main reference for confidence, source-held-out testing and human-review flags |
| **Beyond Classification Accuracy: Uncertainty-Aware Deep Learning for Coral Reef Monitoring** — Mathew Wyatt, Julie Vercelloni, Sharyn M. Hickey, Rebecca Fisher, Manuel Gonzalez-Rivero, Nikolaus Callow, Mohammed Bennamoun and Ben Radford (2026) | Studies uncertainty and its propagation into coral-monitoring estimates | Focuses on uncertainty in ecological quantities, not only frame classification | Advanced methods may exceed six-month scope; requires careful calibration and validation | Justifies reporting uncertainty/variation in composition estimates; use simple bootstrap intervals |
| **Deep Learning for Automated Coral Reef Monitoring: A Novel System Based on YOLOv8 Detection and DeepSORT Tracking** — Younes Ouassine, Noël Conruyt, Mohsen Kayal, Philippe A. Martin, Lionel Bigot, Vignes Lebbe Regine, Hajar Moussanif and Jihad Zahir (2025) | Detects and tracks coral formations in underwater imagery/video | Provides a practical object-detection/tracking baseline | Object boxes are not equivalent to coral cover or health; within-video tracking does not prove cross-year identity | Supporting citation only if tracking is implemented; otherwise not central |
| **A Generalized Machine Learning Model for Long-Term Coral Reef Monitoring in the Red Sea** — Justin J. Gapper, Surendra Maharjan, Wenzhao Li, Erik Linstead, Surya P. Tiwari, Mohamed A. Qurban and Hesham El-Askary (2024) | Applies ML to longer-term/spatially varying Red Sea monitoring | Relevant emphasis on generalization across monitoring conditions | Different geography and sensing context; not a complete underwater-health pipeline | Supports site-held-out and temporal/spatial generalization design |
| **A New Deep Learning Engine for CoralNet** — Qimin Chen, Oscar Beijbom, Stephen Chan, Jessica Bouwmeester and David Kriegman (2021) | Deep-learning engine for point-based CoralNet annotation | Practical route to scalable expert point labels and cover estimates | Sparse points do not provide dense boundaries or temporal relationships | Useful if dense masks are too expensive; supports the annotation protocol |
| **CoralVOS: Dataset and Benchmark for Coral Video Segmentation** — Ziqiang Zheng, Yaofeng Xie, Haixin Liang, Zhibin Yu and Sai-Kit Yeung (2023 preprint) | Dense coral video-object segmentation benchmark | Addresses temporal segmentation rather than isolated image classification | Video segmentation is not the same as reef-health estimation; local transfer is untested | Future video extension and temporal-consistency reference |
| **CoralVOS: Dataset and Benchmark for Dense Coral Video Segmentation** — Ziqiang Zheng, Haixin Liang, Yaofeng Xie, Zhibin Yu and Sai-Kit Yeung (2025 workshop version) | Published/open version of the CoralVOS benchmark | More mature version of the same video-segmentation work | Same limitation: no direct health index or Lakshadweep validation | Cite the published version if discussing video segmentation |
| **Sea-Thru: A Method for Removing Water From Underwater Images** — Derya Akkaynak and Tali Treibitz (2019) | Physical underwater image restoration and colour correction | Provides a principled alternative to generic enhancement | Restoration may introduce artefacts or change biological evidence | Use only as an optional preprocessing experiment |
| **On Calibration of Modern Neural Networks** — Chuan Guo, Geoff Pleiss, Yu Sun and Kilian Q. Weinberger (2017) | Studies why neural-network confidence is often poorly calibrated | Provides the basis for meaningful confidence scores | General image-classification work, not reef/video-specific | Use to justify confidence-versus-correctness evaluation |
| **A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks** — Dan Hendrycks and Kevin Gimpel (2017) | Simple confidence-based error/OOD detection | Transparent baseline for a trust layer | Maximum softmax confidence can fail under strong domain shift | Use as the simplest abstention/OOD comparison |
| **Advancing Coral Reef Monitoring: A Deep Learning Perspective on Automated Segmentation and Classification** — Hafizi Malik, Muhammad Faiz Mohd Hanapiah, Siti Fauziah Toha, Mohd Zaini Mustapa, Aiman Hisyam Azmi, Hazrul Amirul Johari, Ahmad Syahrin Idris, Azhar Mohd Ibrahim, Philippe De Wilde and Amir I. A. Alqedra (2026) | Reviews deep-learning segmentation/classification approaches for reef monitoring | Useful taxonomy of model families and monitoring tasks | Review/perspective; not a project-specific benchmark | Use for background and model-selection discussion, not as a main experimental foundation |

## 4. Synthesis of the literature

The literature falls into five connected groups.

### 4.1 Ecological monitoring and Lakshadweep context

Pandiyarajan et al. and Dey et al. establish why reef monitoring matters in South India and Lakshadweep. Dey et al. show that coral composition varies with depth, wave exposure and recurrent bleaching history. This implies that a computer-vision output must be interpreted alongside survey context rather than presented as an isolated universal health score.

### 4.2 Automated benthic composition

Beijbom-related CoralNet work, Chen et al. and González-Rivero et al. demonstrate the basic workflow of sampling image points, classifying benthic categories and converting predictions into coral-cover estimates. This is the most feasible methodological foundation for the BTP because it does not require complete 3D reconstruction or extensive local labels.

### 4.3 Segmentation and underwater video

Coralscapes, CoralVOS and DeepReefMap show the movement toward dense semantic understanding and video-based reef surveys. They are important for the long-term vision, but their data and technical requirements are broader than the six-month core. The BTP can demonstrate sampled-frame processing without committing to full video-object tracking or 3D mapping.

### 4.4 Coral-condition and disturbance assessment

The two Shao papers provide health-condition categories and modern transfer-learning ideas. Kopecky et al. show how AI and photogrammetry can quantify coral loss after bleaching. Together, they support the health-assessment motivation, but they also show why disease labels, detailed stressors and 3D temporal change should remain optional.

### 4.5 Trustworthy AI

Wyatt et al. establish OOD detection and uncertainty as necessary for coral monitoring. Guo et al. provide the calibration foundation, and Hendrycks and Gimpel provide a transparent OOD baseline. For this BTP, trustworthy AI can be implemented with confidence thresholding, image-quality screening and held-out-source testing rather than a large collection of advanced methods.

## 5. Research gap

The collected papers solve important but mostly separate problems. Some estimate benthic cover, some classify coral condition, some process video, some measure bleaching-related change, and others evaluate OOD detection or uncertainty. Few provide a compact workflow that connects:

1. underwater image prediction;
2. interpretable reef-condition indicators;
3. transfer to a new reef source;
4. confidence and image-quality screening; and
5. a clear statement of what is required for Lakshadweep validation.

The BTP addresses this integration gap at a modest scale.

## 6. Proposed BTP approach from the literature

### 6.1 Input

Use labelled underwater images and, if available, sampled frames from ROV video. Preserve source, site, date, depth, camera and quality metadata.

### 6.2 Labels

Use four broad classes:

- live/healthy-looking coral;
- dead coral/rubble;
- algae/overgrowth;
- sand/rock/other substrate.

These are more robust across datasets than a detailed species or disease taxonomy.

### 6.3 Model

- Baseline: frozen pretrained image backbone with a classification head.
- Main model: one fine-tuned ResNet, EfficientNet or small vision transformer.
- Optional extension: dense segmentation on a small Coralscapes subset.

### 6.4 Reef-condition estimate

For each image or survey group, convert predictions into class proportions. For example:

```text
Live coral: 42%
Dead coral/rubble: 24%
Algae/overgrowth: 18%
Sand/rock/other substrate: 16%
```

### 6.5 Trust layer

- confidence threshold selected on validation data;
- blur/darkness filter;
- held-out-source/site evaluation;
- optional simple maximum-softmax OOD baseline;
- human-review flag for uncertain or unfamiliar samples.

### 6.6 Temporal title component

If compatible repeated survey data exist, compare composition percentages between two groups and report percentage-point differences with bootstrap variation. Do not attempt colony identity, 3D reconstruction or causal change attribution in the six-month implementation.

## 7. Evaluation plan

### Classification

- accuracy;
- macro-F1;
- precision and recall per class;
- confusion matrix.

### Generalization

- random split versus site/source-held-out split;
- performance on a different reef source;
- performance under different image-quality conditions.

### Trustworthiness

- confidence-versus-correctness plot;
- percentage of predictions flagged;
- accuracy before and after abstention;
- qualitative review of flagged images.

### Ecological output

- per-image or per-survey class proportions;
- error against available expert composition estimates;
- sensitivity of estimates to low-quality samples.

## 8. Feasibility for six months

The feasible core is:

1. use Moorea or a selected ReefNet subset for point/patch classification;
2. use Coralscapes only as an optional segmentation or external benchmark;
3. train one baseline and one main model;
4. add confidence and image-quality screening;
5. evaluate on a held-out source;
6. produce broad reef-condition indicators;
7. test Lakshadweep imagery if available, otherwise report transfer limitations.

The following should remain future work: detailed bleaching/disease classification, CoralVOS-style video segmentation, DeepReefMap 3D reconstruction, underwater photogrammetry, colony-level temporal tracking and advanced Bayesian uncertainty.

## 9. Conclusion

The literature supports a focused and defensible BTP. Lakshadweep ecological studies provide the real-world context; CoralNet-style studies provide the composition-estimation formulation; Coralscapes and DeepReefMap provide modern segmentation and video directions; Shao et al. provide condition-classification concepts; Kopecky et al. motivate disturbance-related comparison; and Wyatt et al. provide the trustworthy-AI perspective.

The project should therefore present reef health as a set of visible, measurable indicators rather than a universal score. Its contribution will be a reproducible transfer-learning workflow that estimates broad coral-reef condition, tests performance under source shift and warns when predictions are unreliable.

## Selected local papers

The PDF collection is organized under [ `/mnt/Work/BTP/papers/` ](/mnt/Work/BTP/papers/). Current scope and implementation decisions are in [btp-recommended-approach.md](/mnt/Work/BTP/project/btp-recommended-approach.md), [presentation-architecture-review.md](/mnt/Work/BTP/project/presentation-architecture-review.md), and [implementation-requirements.md](/mnt/Work/BTP/project/implementation-requirements.md).
