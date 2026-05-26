# Week 4 Result Summary - no-NMS AP Validation

## Branch
- `week4-e2e-validation`

## Implemented
- `val_e2e.py`
- `research/week4/compare_metrics.py`

## Validation modes
- no-NMS: confidence filtering + top-k only
- NMS: class-aware NMS reference path

## Outputs
- `runs/val/week4_e2e_no_nms/metrics.json`
- `runs/val/week4_e2e_nms/metrics.json`
- `/content/drive/MyDrive/yolo_v9_v10_e2e_research/week4_e2e_validation/week4_e2e_metric_comparison.csv`

## Interpretation
This week validates that the no-NMS path can produce AP-style metrics.
COCO128 smoke validation is not the final paper result.
