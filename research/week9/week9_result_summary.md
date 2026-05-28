# Week 9 Result Summary - COCO2017 Subset 5k/1k

## Scope

Week 9 uses a deterministic MS COCO 2017 subset with 5,000 train images and 1,000 validation images.

## Motivation

COCO128 is useful for smoke tests, but it is too small for persuasive performance evaluation.
This week improves experimental credibility while remaining feasible in free Colab.

## Compared groups

1. Baseline YOLOv9-S + NMS
2. Proposed V10DualDDetect + NMS
3. Proposed V10DualDDetect + no-NMS

## Dataset

- Source: MS COCO 2017 train/val annotations
- Train subset: 5,000 images
- Validation subset: 1,000 images
- Classes: COCO 80 object classes
- Sampling: deterministic, seed = 42
- Class coverage: balanced first-pass, then random fill

## Interpretation

This is stronger than COCO128, but still not Full COCO.
The paper should describe it as a deterministic COCO2017 subset experiment.
