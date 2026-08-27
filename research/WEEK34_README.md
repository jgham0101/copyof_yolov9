# Week34 — Internal vs COCO API Discrepancy Audit

## Fresh reproduction

Baseline conf=0.1:
- internal mAP50-95: 0.484000
- COCO API AP50-95: 0.427000

Proposed conf=0.1:
- internal mAP50-95: 0.480000
- COCO API AP50-95: 0.291000

## Canonical conf=0.001

Baseline:
- internal mAP50-95: 0.447000
- COCO API AP50-95: 0.448000

Proposed:
- internal mAP50-95: 0.410000
- COCO API AP50-95: 0.410000

## JSON structural integrity

Baseline:
- structural integrity: True
- fatal structural errors: 0
- zero-area bbox telemetry: 1

Proposed:
- structural integrity: True
- fatal structural errors: 0
- zero-area bbox telemetry: 0

## Direct COCO audit

Baseline aware100 / aware300 / agnostic100:
- 0.426944
- 0.426950
- 0.464565

Proposed aware100 / aware300 / agnostic100:
- 0.291465
- 0.291465
- 0.310042

## Root-cause classification

CONFIDENCE_PREFILTER_SENSITIVITY

## Original strict Problem 1 gate

- internal retention: 0.991736
- internal gap: 0.004000
- COCO API retention: 0.682677
- COCO API gap: 0.135479

## Final status

- Problem 1: RESIDUAL_REMAINS
- Problem 2 allowed: False
- Next: REVIEW_WEEK34_AUDIT_BEFORE_PROBLEM2

No acceptance gate or threshold was relaxed.
