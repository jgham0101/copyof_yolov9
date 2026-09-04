# Week37 — O2O TP/FP Score Separation & PR Ranking Residual Diagnostic

## Scope
- Training: NO
- Checkpoint/model/loss/assigner modification: NO
- no-NMS: NO
- Input: Week36 20e canonical conf=.001 + native-NMS prediction JSON

## COCO reproduction
- Baseline AP50-95: 0.454488
- Proposed AP50-95: 0.422504
- Gap: 0.031984

## TP/FP score separation @ IoU=0.50
- Baseline separation AUC: 0.952417
- Proposed separation AUC: 0.934381
- AUC gap B-P: 0.018035
- Baseline TP median: 0.604965
- Proposed TP median: 0.360370
- Baseline FP q95: 0.092700
- Proposed FP q95: 0.056420

## TP/FP score separation @ IoU=0.75
- Baseline separation AUC: 0.952941
- Proposed separation AUC: 0.935600
- AUC gap B-P: 0.017342
- Baseline TP median: 0.703130
- Proposed TP median: 0.494410
- Baseline FP q95: 0.107600
- Proposed FP q95: 0.070350

## FN decomposition @ IoU=.50
- NO_LOCALIZED_CANDIDATE: Baseline 9.5858% / Proposed 10.6757% (P-B +1.0899%)
- WRONG_CLASS_LOCALIZATION: Baseline 2.2650% / Proposed 2.5568% (P-B +0.2917%)
- SAMECLASS_CANDIDATE_UNMATCHED: Baseline 0.5642% / Proposed 0.6165% (P-B +0.0523%)

## PR region gaps
- IoU 0.50 low_recall: precision gap B-P = 0.025067
- IoU 0.50 mid_recall: precision gap B-P = 0.049671
- IoU 0.50 high_recall: precision gap B-P = 0.026893
- IoU 0.75 low_recall: precision gap B-P = 0.038776
- IoU 0.75 mid_recall: precision gap B-P = 0.050318
- IoU 0.75 high_recall: precision gap B-P = 0.019239

## Final
- Root cause: **RESIDUAL_NOT_LOCALIZED**
- Active flags: none
- Problem 1: **RESIDUAL_REMAINS**
- Problem 2: **BLOCKED**
- Next: **WEEK38_REVIEW_RESIDUAL_AND_MINIMAL_ONE_FACTOR_ABLATION**

Diagnostic thresholds are heuristics only. No statistical significance is claimed and the strict Problem 1 gate was not relaxed.

## Provenance
- Week36 base commit: `d61e96b37a378848540e5bb40e3f95cc18c51992`
- Baseline prediction SHA256: `95450a55bb3e54aa388290da95ddb820df340509e06a1437940f4aa24c0c49e3`
- Proposed prediction SHA256: `1fda458dac6411d31b3789df7e68c88d154e3aed642e52126679b028c887b500`
- Week37 training: **NO**
- Week37 model/head/loss/assigner modification: **NO**
- Week37 no-NMS: **NO**
