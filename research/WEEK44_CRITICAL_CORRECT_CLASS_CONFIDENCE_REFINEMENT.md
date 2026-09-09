# Week44 — Critical Correct-Class Confidence Refinement

- Week43 dominant confidence failure: 57.83%
- Intervention: O2O positive target-conditioned confidence-up auxiliary
- lambda: 1.0
- Week43 validation IDs used for training: False

## AP50-95
- Baseline internal: 0.453000
- Week39 internal: 0.429000
- Week44 internal: 0.428000
- Week44-Week39 internal: -0.001000
- Baseline COCO API: 0.454000
- Week39 COCO API: 0.428000
- Week44 COCO API: 0.427000
- Week44-Week39 COCO API: -0.001000

## Week43 critical rescue
- Overall: 250/849 (29.45%)
- Confidence failure: 113/491 (23.01%)

## Strict Problem 1 gate
- Internal retention: 94.4812% / False
- Internal gap: 0.025000 / False
- COCO retention: 94.0529% / False
- COCO gap: 0.027000 / False

- Verdict: **CRITICAL_CONFIDENCE_REFINEMENT_NOT_SUFFICIENT**
- Problem 1: **RESIDUAL_REMAINS**
- Problem 2 allowed: **False**
- Next: **REASSESS_DISTRIBUTED_RESIDUAL_STRATEGY**

Strict gate was not relaxed. No statistical significance claim.

## Provenance
- Base branch: `week43-ap-critical-failure-subset-tracing`
- Base commit: `4836f2db193b6453b98066cae132186dca5349c6`
- Proposed init SHA256: `b7ca64c7f61bafc10ee883f84dae1f8facf0cfd13c79ff0e0e9f0d33333b5ce3`
- Treatment SHA256: `ed1c895c2a644f86210e2e8ab6015273be55a3e4156735097843f97874783b62`
- Validation critical IDs used for training: `False`
