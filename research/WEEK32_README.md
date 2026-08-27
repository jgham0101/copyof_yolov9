# Week32 — O2O Confidence Operating-Point Finalization

Week31 found:
- O2M+NMS @ conf=0.001: mAP50-95 0.443
- O2O+NMS @ conf=0.001: mAP50-95 0.410
- O2O+NMS @ conf=0.010: mAP50-95 0.432

Week32 performs no training.

Goals:
1. bracket the O2O+NMS confidence operating-point optimum beyond 0.01,
2. reject an upper-bound artifact,
3. evaluate O2M at the exact same finalized threshold,
4. decide whether Problem 1 can proceed to the final three-model confirmation.

PASS gate:
- O2O peak is bracketed,
- O2O best AP retains >=98% of canonical O2M AP,
- at the same threshold O2O retains >=98% of O2M,
- same-threshold absolute AP gap <=0.010,
- at least one adjacent threshold is within 0.005 AP of the best value.

Even if PASS, Problem 2 remains blocked.
The next stage is the unified Problem-1 three-model final confirmation.
