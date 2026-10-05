# Dataset Plan

## Required datasets

| Priority | Dataset | Expected role | Selection rule |
|---:|---|---|---|
| 1 | Coralscapes | Main broad-benthic semantic segmentation dataset: 2,075 images, 39 classes and dense masks. | Use the official release; preserve the original split and site metadata. |
| 2 | NOAA PIFSC coral-bleaching dataset | Coral-condition classifier: visible healthy/bleached labels. | Audit class definitions, licence, duplicates and image provenance before training. |
| 3 | CoralNet, ReefNet or MERMAID - choose one | A later, external-source robustness test. | Download only the minimum licensed subset needed for a sealed test. |
| 4 | Your own small demonstration set | Qualitative demo and reviewer assessment. | Keep fully separate from every training/validation decision. |

## Annotation requirements

- **Segmentation:** one single-channel PNG class-ID mask per image; record ignored pixels as 255. This matches YOLO26 semantic-segmentation input requirements.
- **Condition classifier:** crop-level `healthy-looking`, `bleached-looking`, `dead-rubble`, and `uncertain` labels with an annotation guide and reviewer/source ID.
- **Metadata:** unique image ID, source, site/dive group where available, date, resolution, licence and split. No metadata is optional for evaluation integrity.

## Data quantities

- Coralscapes is enough for a serious segmentation core when split by site, even without downloading additional giant archives.
- For a condition proof of concept, target at least 300 reviewed crops per retained class; 500+ per class is preferable. Remove or retain ambiguity only according to the written label policy - never silently relabel it.
- Reserve 15-20% of grouped data for a final test set and 10-15% for validation. Exact proportions matter less than preventing location/source leakage.

## Dataset decisions to avoid

- Do not merge datasets until their labels are mapped and visually audited.
- Do not use images whose licence or redistribution terms are unclear.
- Do not use a train/test split across frames from the same video or patches from the same panorama.
- Do not call a public-dataset result “real-world coral health accuracy.”
