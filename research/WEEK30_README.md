# Week30 — O2O Box-vs-Classification Inference Quality Gap Diagnostic

Week29 stopped because O2O+NMS retained only 92.55% of O2M+NMS mAP50-95.

Week30 performs no training and no no-NMS experiment.

Four evaluation-only variants are built from the same Week28-v2 DUAL checkpoint:
- M_M: O2M box + O2M cls
- M_O: O2M box + O2O cls
- O_M: O2O box + O2M cls
- O_O: O2O box + O2O cls

All use the same native YOLOv9 decode and NMS.

A 500-image pre-NMS candidate probe compares:
- best spatial IoU
- IoU50/IoU75 candidate coverage
- correct-class IoU coverage
- GT-class score at the spatial-best candidate
- top-class correctness

The next intervention is chosen only from this decomposition.
Problem 2/no-NMS work remains blocked until Problem 1 is resolved and the
three-model final confirmation is completed.
