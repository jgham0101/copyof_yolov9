# Week48 — O2O Uniqueness-Consistency Training Intervention

## Design
- Control: Week39 cls×2 fresh-20e frozen control
- Treatment: same recipe + O2O hardest-overlap unmatched class-negative BCE auxiliary
- Duplicate IoU threshold: 0.70
- Hard negative: highest same-target-class logit among unmatched overlapping anchors
- Auxiliary gradient: O2O classification only
- Treatment: fresh 20e from identical Week33 proposed initialization
- Control retraining: False
- Statistical significance claimed: False

## Week47 causal basis
- C70 rescue fraction: 0.9679
- K500-K300 AP: +0.000263
- GT multiplicity>=3: 0.5687

## Control
- +NMS COCO AP50-95: 0.428112
- no-NMS COCO AP50-95: 0.410260
- Problem2 gap: 0.017852
- same-class duplicate pairs/image: 256.4206
- multiplicity>=3 fraction: 0.5687

## Treatment
- +NMS COCO AP50-95: 0.424963
- no-NMS COCO AP50-95: 0.409553
- Problem2 gap: 0.015410
- same-class duplicate pairs/image: 252.9806
- multiplicity>=3 fraction: 0.5403

## Treatment - Control
- +NMS AP: -0.003149
- no-NMS AP: -0.000708
- Problem2 gap change: -0.002441
- duplicate reduction fraction: 1.3415%
- multiplicity>=3 reduction fraction: 4.9848%

## Gate
- Verdict: **DUPLICATE_HARD_NEGATIVE_AUX_NOT_CAUSALLY_SUFFICIENT**
- Next: **WEEK49_REVISIT_UNIQUENESS_CONSISTENCY_MECHANISM**

Problem 1 remains **ANALYSIS_COMPLETED_WITH_ACCEPTED_RESIDUAL**.
The strict Problem 1 gate is not relaxed.

## Provenance
- Week47 base commit: `ec0afbfbfc9a561c5a8b84437e8923605bbf9e78`
- Proposed init SHA256: `b7ca64c7f61bafc10ee883f84dae1f8facf0cfd13c79ff0e0e9f0d33333b5ce3`
- Treatment raw SHA256: `87fd29215f2616ff966e06a1b2ac57569cfd9fc29b295eb1bad8a5f4bf6e6568`
- Treatment proxy SHA256: `ba31c269c2d43aed49e4f27d99baf706951d55608d50d842795d49606ff33dac`
- One factor: hardest unmatched O2O same-target-class negative BCE at predicted-box IoU>=0.70.
