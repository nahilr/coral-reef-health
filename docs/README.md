# BTP Project Index

## Current project

**Trustworthy AI-Assisted Coral Reef Health Assessment**

This phase targets coral-reef imagery generally. It uses CoralscapesV2 for visible benthic composition and coral-condition labels, with explicit quality and reliability warnings. Video is analyzed by sampled frames; temporal colony tracking and repeat-survey change detection are outside the current implementation scope.

## Current decisions and implementation

- [Recommended approach](btp-recommended-approach.md) - scope, system flow, dataset/model choices, trust contract, feasibility, and reproducibility requirements.
- [Dataset inventory and plan](dataset-inventory.md) - local availability, contents, annotations, source links, and planned roles/constraints.
- [Paper inventory](paper-inventory.md) - direct review of papers and presentation selections, with current-implementation fit and metric caveats.
- [Literature review](literature-review.md) - narrative background, synthesis, and research gap.
- [Verified sources](research-sources.md) - dataset and technical documentation links.

## Archived and consolidated notes

- Earlier standalone paper-fit/comparison notes, dataset compatibility, architecture review, feasibility assessment, and implementation requirements were consolidated into the three current documents above. Their original copies remain under [`archive/`](archive/) for traceability; use the current documents as authoritative.
- Older research notes remain under [`archive/`](archive/) when they contain source analysis that is not duplicated in the current decision documents.

The current decision documents above distinguish the planned BTP scope from the implemented pipeline. When scope or implementation details differ, use the implementation status in the paper inventory and current code as the factual baseline.
