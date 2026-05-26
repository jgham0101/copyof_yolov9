# Week 4 - no-NMS AP Validation

## Goal

Build a validation loop that computes AP/mAP-style metrics for the one-to-one branch without NMS.

## Added files

- `val_e2e.py`
- `research/week4/compare_metrics.py`

## Modes

- `--postprocess no-nms`: confidence filtering + top-k, no IoU suppression
- `--postprocess nms`: class-aware NMS reference path

## Scope

COCO128 smoke validation confirms that the AP evaluation loop works. It is not the final paper-scale result.
