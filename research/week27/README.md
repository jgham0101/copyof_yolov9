# Week27 — Active O2O Forward Equivalence Audit

Base:
- week26-dormant-o2o-equivalence

Single added factor:
- execute O2O bbox/classification heads on detached copies of the SAME [15,18,21] features.

Still disabled:
- O2O assignment
- O2O loss
- optimizer/training
- O2O inference selection
- no-NMS

Evaluation remains:
- O2M + native NMS

Required evidence:
1. O2O forward is actually executed.
2. O2O-only backward gives gradient to O2O head but not to O2M/body/input.
3. Week26 and Week27 O2M feature/raw/decode/NMS remain equivalent.
4. At deepcopy initialization, Week27 O2M and O2O raw/decode remain equivalent.
5. COCO val2017 5k mAP50-95 retention >= 98%.

Only after PASS may Week28 activate O2O assignment/loss.
