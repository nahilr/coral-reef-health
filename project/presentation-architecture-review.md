# Architecture Review - Revised Scope

## Decision

The original architecture is worth retaining, with two changes: remove the location-specific and temporal modules, and make reliability a mandatory gate rather than a final add-on.

## Revised flow

```text
Upload image/video
 -> validate type, resolution and metadata
 -> sample video frames / quality gate
 -> broad benthic segmentation
 -> coral crop extraction
 -> visible-condition classification
 -> confidence, calibration and OOD checks
 -> accepted indicators or "needs expert review"
 -> overlay + report
```

## Implementation rules

| Original block | Revised decision |
|---|---|
| Frame extraction | Keep for video; sample at a fixed rate and deduplicate similar frames. |
| Underwater preprocessing | Keep original pixels and treat enhancement as an optional, measured experiment. |
| Coral/benthic detection & segmentation | Make this the first trained model and limit it to broad classes supported by masks. |
| Coral condition assessment | Make it a separate classifier on coral crops, never a guessed output from Model A. |
| Quantitative health indicators | Report composition and visible-condition indicators with denominators and reliability state. |
| Temporal analysis | Remove from the current scope. |
| Cumulative report/dashboard | Keep as final integration only. |

## Critical design choice

Use semantic segmentation, not object detection, for the core health-assessment proxy: the useful output is the **area** occupied by coral, algae, rubble and substrate. Bounding boxes are poorly aligned with cover estimates. If individual coral boundaries are needed later, add instance segmentation as a separately evaluated extension.

## Trust contract

Every result must include model version, dataset version, quality flags, confidence/uncertainty state, accepted-pixel/crop count and a visible `needs review` state. This is what makes the project trustworthy rather than merely an image classifier.
