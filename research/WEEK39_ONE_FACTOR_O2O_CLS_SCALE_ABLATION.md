# Week39 — One-Factor O2O Classification-Loss Scale Ablation

## Intervention
- Control: Week36 O2O cls scale = 1.0
- Treatment: Week39 O2O cls scale = 2.0
- Fresh 20e from identical Proposed initialization
- All non-classification variables fixed
- no-NMS: NO

## Canonical AP50-95
- Baseline COCO AP50-95: 0.454000
- Week36 Control Proposed: 0.423000
- Week39 Treatment: 0.428000
- Treatment - Control: +0.005000

## Confidence-prefilter stage
- Baseline: 593
- Control: 1245
- Treatment: 1214
- Excess recovery fraction: 0.04754601226993865

## Correct-class score/classification drop
- Baseline: 1260
- Control: 2115
- Treatment: 2093
- Excess recovery fraction: 0.025730994152046785

## Strict Problem 1 gate
- Internal retention: 94.7020% / pass=False
- Internal gap: 0.024000 / pass=False
- COCO retention: 94.2731% / pass=False
- COCO gap: 0.026000 / pass=False

## Final
- Verdict: **PARTIAL_CAUSAL_SUPPORT**
- Problem 1: **RESIDUAL_REMAINS**
- Problem 2 allowed: **False**
- Next: **WEEK40_TARGET_QUALITY_AND_CORRECT_CLASS_CONFIDENCE_DIAGNOSTIC**

Single-seed one-factor ablation; no statistical significance is claimed and the strict Problem 1 gate was not relaxed.

## Provenance
- Week38 base commit: `bd9266b56b9608201036159f607086a90decf25b`
- Proposed init SHA256: `b7ca64c7f61bafc10ee883f84dae1f8facf0cfd13c79ff0e0e9f0d33333b5ce3`
- Treatment last SHA256: `f766c69e067981fe24beed0851637710a28e6797b5dd4d5818753faf3eb41586`
- One factor: O2O classification BCE scale 1.0 -> 2.0
