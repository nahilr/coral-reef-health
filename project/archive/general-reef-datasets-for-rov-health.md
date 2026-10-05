# General-Reef Datasets for ROV/Underwater Health Assessment

Checked 2026-09-16.

## Best matches

| Dataset | Best use | Labels/content | Fit for ROV-health project |
|---|---|---|---|
| Coralscapes | Semantic segmentation and cover estimation | 2,075 images, 39 benthic classes, 174,077 expert masks; includes live, dead, bleached, algae, rubble, sand and other classes | Best starting point for pixel-level benthic assessment |
| ReefNet 1.0 | Coral/genus and benthic classification | 32k+ images from 400+ sites with point annotations and geographic metadata | Best for classification and geographic generalization |
| MERMAID community image data | Large-scale point classification | Public coral images with confirmed annotations and survey context | Best for scale and human-in-the-loop workflows |
| NOAA CoralNet Bleaching Classifier package | Bleaching classification | Human point annotations, image metadata, labelset, and imagery from Hawaiian surveys | Best for a focused healthy-vs-bleached experiment |
| Moorea LTER labeled corals | Baseline benthic classification | 2,055 images and about 400,000 point annotations across nine classes | Good local baseline already collected, but not dense health masks |
| DeepReefMap data/pipeline | Video transects and semantic mapping | Large Red Sea video-survey workflow with 39-class semantic mapping | Best methodological reference for video, not the easiest first dataset |

## Recommended combination

Use three datasets rather than trying to train one model on everything:

1. **Coralscapes** for segmentation.
2. **ReefNet or MERMAID** for point-based benthic classification.
3. **NOAA bleaching data** for optional bleaching classification.

Then test trustworthiness by holding out entire sources, locations, or habitats as OOD data.

## Important limitation

Most public reef datasets contain images or photoquadrats, not continuous ROV video. They can train the frame-level model used inside an ROV pipeline, but a genuine video system still needs a video dataset or local video for frame sampling, tracking, temporal aggregation, and transect-level evaluation.

## Sources

- Coralscapes: https://openaccess.thecvf.com/content/ICCV2025W/CVAUI%20%26%20AAMVEM/papers/Sauder_The_Coralscapes_Dataset_Semantic_Scene_Understanding_in_Coral_Reefs_ICCVW_2025_paper.pdf
- ReefNet: https://reefnet-project.github.io/reefnet-2025/ and https://huggingface.co/datasets/ReefNet/ReefNet-1.0
- MERMAID open data: https://registry.opendata.aws/coralreef-image-classification-training/
- NOAA bleaching package: https://www.fisheries.noaa.gov/inport/item/67962
- NOAA CoralNet classifier: https://coralnet.ucsd.edu/source/2947/
- Moorea LTER: https://mcr.lternet.edu/data/datasets/mcr-lter-coral-reef-computer-vision-moorea-labeled-corals
- DeepReefMap: https://www.nature.com/articles/s41598-025-20795-z
