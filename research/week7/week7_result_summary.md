# Week 7 Result Summary - Extended Controlled Experiment

## Scope

Week 7 extends the Week 5 comparison from 1 epoch to 3 epochs under the same COCO128 / image size 320 condition.

## Compared groups

1. Baseline YOLOv9-S + NMS
2. Proposed V10DualDDetect + NMS
3. Proposed V10DualDDetect + no-NMS

## Main outputs

- `week7_experiment_comparison.csv`
- `week7_experiment_comparison.json`
- `week5_week7_combined_comparison.csv`
- `week7_accuracy_table.csv`
- `week7_latency_table.csv`
- `week7_proposed_postprocess_comparison.csv`
- `week7_postprocess_speedup_summary.csv`
- `week7_extended_tables.md`

## Interpretation

This is not a full COCO result. The purpose is to confirm that the comparison pipeline remains stable under a longer controlled training condition.
