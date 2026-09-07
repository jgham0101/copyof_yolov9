# Week38 — Pre-NMS Raw Candidate Rescue Diagnostic

## Scope
- Training: NO
- Model/head/loss/assigner/checkpoint modification: NO
- no-NMS: NO
- canonical conf=.001 / NMS IoU=.7 / max_det=300

## Post-NMS NO_LOCALIZED reproduction
- Baseline: 9.561%
- Proposed: 10.673%

## Rescue among post-NMS misses
- Baseline raw rescue: 1105 / 3474 (31.808%)
- Proposed raw rescue: 1639 / 3878 (42.264%)
- Baseline conf-qualified rescue: 14.738%
- Proposed conf-qualified rescue: 10.160%

## Stage gaps Proposed - Baseline
- RAW_CANDIDATE_MISS: count gap -130, rate gap -0.3578%, positive-gap share 0.00%
- CONF_PREFILTER_DROP: count gap +652, rate gap +1.7944%, positive-gap share 100.00%
- NMS_OR_MAXDET_DROP: count gap -118, rate gap -0.3248%, positive-gap share 0.00%

## Near-threshold raw IoU among post-NMS misses
- Baseline raw IoU .45-.50: 13.155%
- Proposed raw IoU .45-.50: 11.630%

## Final
- Root cause: **CONFIDENCE_PREFILTER_RESIDUAL**
- Dominant positive stage: **CONF_PREFILTER_DROP**
- Problem 1: **RESIDUAL_REMAINS**
- Problem 2: **BLOCKED**
- Week39: **WEEK39_ONE_FACTOR_CONFIDENCE_TARGET_OR_LOSS_SCALE_ABLATION**

Routing thresholds are diagnostic heuristics only. No statistical significance is claimed and the strict Problem 1 gate was not relaxed.

## Provenance
- Week37 base commit: `3fe60ac4982368c42a96e79589a789265339e885`
- Baseline e20 SHA256: `c50310277af9c7eb8f951fe86787c15d793076ef7f60be853927a827e6ce2a61`
- Proposed e20 O2O proxy SHA256: `8668e126041e8a2e84945bdd9ce01f4894ad12a780dd5f73a25f5a83fbe77c93`
- Week38 training/model/loss/assigner modification: **NO**
- Week38 no-NMS: **NO**
