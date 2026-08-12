# Week 19 — Same-feature Dual Head / Gradient Path Audit

## Purpose
Week19 verifies whether the current split-neck O2M/O2O design is a structural
cause of weak scratch convergence.

## Controlled variants
A: current split feature, O2M weight 0.25
B: same feature only
C: same feature + O2O copy initialization
D: same feature + copy initialization + O2M weight 1.0

## Rule
Do not add DeFCN/PSS-style modules until the YOLOv10-style basic
dual-assignment pipeline has been mechanically verified.
