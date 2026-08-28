# Week35 — O2O Confidence / Recall / Ranking Residual Diagnostic

## Scope
- Training: **NO**
- Model/checkpoint modification: **NO**
- no-NMS: **NO**
- Input: Week34 canonical `conf=0.001 + native NMS` prediction JSON

## Canonical AP reproduction
- Baseline COCO AP50-95: **0.448166**
- Proposed COCO AP50-95: **0.410395**
- Absolute gap: **0.037771**
- Retention: **91.5721%**

## GT coverage / localization
Baseline any/same-class IoU50 coverage: 90.4335% / 88.0391%
Proposed any/same-class IoU50 coverage: 89.0133% / 85.8979%

Baseline median best-any / same-class IoU: 0.857997 / 0.846520
Proposed median best-any / same-class IoU: 0.849219 / 0.838803

## Confidence
Baseline median TP50 score / survival@.1: 0.583120 / 70.9123%
Proposed median TP50 score / survival@.1: 0.105420 / 44.1971%

## Ranking
Baseline Spearman(score, same-class IoU): 0.280286
Proposed Spearman(score, same-class IoU): 0.284038

Baseline TP50 top-10 image/all-GT: 56.6286%
Proposed TP50 top-10 image/all-GT: 54.8534%

## Root-cause classification
**CONFIDENCE_CALIBRATION_RESIDUAL**

Active flags:
- CONFIDENCE_CALIBRATION_RESIDUAL

## Current status
- Problem 1: **RESIDUAL_REMAINS**
- Problem 2: **BLOCKED**
- Next: **WEEK36_LONGER_TRAINING_CONVERGENCE_TEST_3E_10E_20E**

Diagnostic thresholds are heuristics only. No Problem 1 acceptance threshold was relaxed.


## Provenance
- Baseline prediction SHA256: `17aff221bd9fee1022166cdc17242d236b0b5878c2ec0b0f201a33033b70400a`
- Proposed prediction SHA256: `7f091e70a591fbab5c7375abe9c150b5e512040b8edf38779dd00dcdb9ca95d7`
- Training: **NO**
- no-NMS: **NO**
- Model source modified: **NO**
