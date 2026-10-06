Title: Trustworthy AI-Assisted Coral Reef Health Assessment

## Project Focus & Scope
The project develops an AI-assisted framework for automated benthic composition and coral reef health assessment from underwater imagery and video surveys.

### Core Objectives:
1. **Automated Benthic Segmentation**: Automatically segment coral and key benthic substrate categories (live coral, bleached coral, dead coral, algae, rubble, sand/rock, and other).
2. **Quantitative Reef Health Indicators**: Compute standardized marine ecological indicators:
   - Live Coral Cover (LCC %)
   - Bleaching Ratio (%)
   - Coral Mortality Ratio (%)
   - Macroalgae Competition Index (Algae-to-Coral ratio)
   - Integrated Reef Health Tier (Excellent / Good / Fair / Poor-Degraded)
3. **Trustworthy AI Gating**: Implement a rigorous review and abstention gate:
   - Image quality validation (blur, under/over-exposure, low contrast, severe color cast)
   - Prediction uncertainty estimation (entropy over detection confidence scores)
   - Expected Calibration Error (ECE) measurement
   - Transparent acceptance state (`ACCEPTED` vs `NEEDS EXPERT REVIEW`)

### Scope Note:
Site-specific Lakshadweep field validation and multi-year temporal change detection have been explicitly scoped out of the current Phase 1 implementation to establish a robust, verified transfer baseline and reproducible AI pipeline on benchmark data.

- Potential partnership with eyerov


## Sources
[datasets - Google Drive](https://drive.google.com/drive/u/5/folders/1sh7KEuhbBaVNr5g7JNwgBpwvz_DY8lFn)

[Occurrence records of coral genera from long-term photoquadrat monitoring at three atolls of the Lakshadweep Archipelago, India (2001-2022)](https://www.gbif.org/dataset/36fbf9a0-1c96-4080-8bba-06b89f444257)

[MCR LTER: Coral Reef: Computer Vision: Moorea Labeled Corals | Moorea Coral Reef LTER](https://mcr.lternet.edu/data/datasets/mcr-lter-coral-reef-computer-vision-moorea-labeled-corals)

[Dryad | Data: Local environment and coral composition affect recovery and determine long-term coral responses to recurrent mass mortalities in the Lakshadweep Archipelago](https://datadryad.org/dataset/doi%3A10.5061/dryad.vq83bk3z5)

[CoralNet](https://coralnet.ucsd.edu/)

[IEEE Xplore](https://ieeexplore-ieee-org-iiitkottayam.knimbus.com/Xplore/home.jsp)

[allencoralatlas.org](https://allencoralatlas.org/)

[Lakshadweep, India Regional Products 2026-08-30](https://coralreefwatch.noaa.gov/product/vs/gauges/lakshadweep.php)

[[2503.23012] Multi-label classification for multi-temporal, multi-spatial coral reef condition monitoring using vision foundation model with adapter learning](https://arxiv.org/abs/2503.23012)

[Rapid consistent reef surveys with DeepReefMap | Scientific Reports](https://www.nature.com/articles/s41598-025-20795-z)

[ReefNet](https://reefnet-project.github.io/reefnet-2025/)

[ReefNet/ReefNet-1.0 · Datasets at Hugging Face](https://huggingface.co/datasets/ReefNet/ReefNet-1.0?utm_source=chatgpt.com)

[coral-reef-dataset](https://www.kaggle.com/datasets/jxwleong/coral-reef-dataset)
