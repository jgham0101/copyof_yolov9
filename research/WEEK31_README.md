# Week31 — O2O Classification Calibration vs Discriminability Diagnostic

Week30 localized the residual Problem-1 gap to the O2O classification branch.

Week31 performs no training.

Stage A:
Candidate-level scale/ranking probe on fixed 500 COCO val images:
- GT-class score/logit
- wrong-class score/logit
- class rank@1/rank@5
- positive-vs-hard-background score/logit margin
- positive-wins-hard-background rate

Stage B:
Full COCO val2017 threshold sweep for O2O+native-NMS:
- 3e-5
- 1e-4
- 3e-4
- 1e-3 (reused Week30 reference)
- 3e-3
- 1e-2

Interpretation:
- AP recovery from lower confidence threshold + preserved ranking
  -> calibration/score-scale dominant
- little AP recovery + degraded positive/background ranking
  -> discriminability dominant
- intermediate result -> mixed

Problem 2 remains blocked until Problem 1 is solved and the final unified
three-model confirmation is completed.
