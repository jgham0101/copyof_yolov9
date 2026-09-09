# Week40 — O2O Target Quality & Correct-Class Confidence Diagnostic

## Scope
- Training/backward: NO
- Checkpoint/model/head/loss/assigner modification: NO
- Data: canonical COCO train5k seed42, augmentation disabled for paired diagnostic
- Compared: Week36 Control vs Week39 cls×2 Treatment

## Control
- O2M GT coverage: 97.1042%
- O2O GT coverage: 96.9780%
- Coverage gap O2M-O2O: 0.1262%
- O2M per-GT target median: 0.901966
- O2O per-GT target median: 0.880012
- O2O/O2M target ratio median: 0.990246
- O2M pred correct median: 0.541080
- O2O pred correct median: 0.313070
- O2O/O2M pred-correct ratio median: 0.764993
- O2O target↔prediction Spearman: 0.563090
- Fraction O2O target ≤ 0.5× O2M: 0.1410%

## Treatment
- O2M GT coverage: 97.1067%
- O2O GT coverage: 96.9434%
- Coverage gap O2M-O2O: 0.1634%
- O2M per-GT target median: 0.902329
- O2O per-GT target median: 0.878710
- O2O/O2M target ratio median: 0.989880
- O2M pred correct median: 0.537700
- O2O pred correct median: 0.379466
- O2O/O2M pred-correct ratio median: 0.878308
- O2O target↔prediction Spearman: 0.595441
- Fraction O2O target ≤ 0.5× O2M: 0.1462%

## Week39 cls×2 effect
- O2O per-GT target median change: -0.001302
- O2O predicted-correct median change: +0.066397
- O2O/O2M target-ratio change: -0.000365
- O2O/O2M predicted-ratio change: +0.113315

## Final
- Root cause: **TARGET_QUALITY_NOT_PRIMARY**
- Active primary flags: none
- Problem 1: **RESIDUAL_REMAINS**
- Problem 2: **BLOCKED**
- Week41: **WEEK41_MINIMAL_INTERVENTION_AFTER_TARGET_QUALITY_REVIEW**

Diagnostic thresholds are routing heuristics only. No statistical significance is claimed and the strict Problem 1 gate was not relaxed.

## Provenance
- Week39 base commit: `36954400c18497231a2224641fef2f1fcf89f995`
- Control SHA256: `eaed181091a0b7bcca4447a95bd6eacadcd32ff387d755152a08cd032128456f`
- Treatment SHA256: `f766c69e067981fe24beed0851637710a28e6797b5dd4d5818753faf3eb41586`
- Training/backward: **NO**
- Assigner/source modification: **NO**
