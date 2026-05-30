# Week 11 - Branch Diagnosis

This week diagnoses why the proposed V10DualDDetect model shows lower mAP than baseline.

Main comparisons:
1. baseline YOLOv9-S + NMS
2. proposed V10DualDDetect native output through val_dual.py
3. proposed one-to-one + NMS through val_e2e.py
4. proposed one-to-one + no-NMS through val_e2e.py

The purpose is to separate:
- NMS-removal loss
- one-to-one branch quality
- proposed head/loss integration issue
