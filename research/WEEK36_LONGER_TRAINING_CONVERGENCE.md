# Week36 — Longer-Training Convergence Test

## Protocol
- Fresh 20e trajectories; no Week33 3e resume
- snapshots: 3/10/20e from same total-20e scheduler
- COCO train5k seed42 / val2017 5k
- canonical conf=.001 + native NMS
- no-NMS: NO

## Epoch 3
- Baseline COCO AP50-95: 0.452000
- Proposed COCO AP50-95: 0.418000
- gap: 0.034000
- retention: 92.4779%
- TP50 median score B/P: 0.586680 / 0.118960
- TP50 score ratio P/B: 0.2028

## Epoch 10
- Baseline COCO AP50-95: 0.454000
- Proposed COCO AP50-95: 0.415000
- gap: 0.039000
- retention: 91.4097%
- TP50 median score B/P: 0.590730 / 0.291720
- TP50 score ratio P/B: 0.4938

## Epoch 20
- Baseline COCO AP50-95: 0.454000
- Proposed COCO AP50-95: 0.423000
- gap: 0.031000
- retention: 93.1718%
- TP50 median score B/P: 0.606130 / 0.363280
- TP50 score ratio P/B: 0.5993

## Final
- Verdict: **MIXED_OR_OTHER_CONVERGENCE_PATTERN**
- Problem 1: **RESIDUAL_REMAINS**
- Problem 2 allowed: **False**
- Next: **WEEK37_REVIEW_CONVERGENCE_AND_MINIMAL_CONFIDENCE_INTERVENTION**

Week36 epoch-3 is not required to reproduce Week35 because it belongs to a fresh total-20e scheduler trajectory.
No statistical significance is claimed and the strict Problem 1 gate was not relaxed.

## Provenance
- Base commit: `fae9f465564df75f6edf201f37bb9f503eca424a`
- Baseline init SHA256: `e232b9c0423ff04608d973e684f8ee808f933e6b7fc052787efb30e43b8acac5`
- Proposed init SHA256: `b7ca64c7f61bafc10ee883f84dae1f8facf0cfd13c79ff0e0e9f0d33333b5ce3`
- Week36 model/loss/assigner architecture modification: **NO**
