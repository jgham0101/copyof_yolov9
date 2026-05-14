# Week 1 - YOLOv9/v10 Dual Head Research Baseline

This folder records the Week 1 setup for the YOLOv9-to-YOLOv10 Dual Head research.

## Repository policy
- Baseline code source: `jgham0101/copyof_yolov9`
- YOLOv10 reference source: `jgham0101/copyof_yolov10`
- Large files such as datasets, runs, and weights are stored in Google Drive, not GitHub.

## Week 1 goal
1. Verify YOLOv9 baseline train/validation on Colab.
2. Identify YOLOv9 head/loss/inference modification points.
3. Identify YOLOv10 one2many/one2one reference implementation points.
4. Prepare Week 2 implementation of `V10DualDDetect`.

## Planned modification target
- `models/yolo.py`
- `utils/loss_tal_dual.py`
- `train_dual.py`
- `val_dual.py`
- `detect_dual.py`

## Do not commit
- `runs/`
- `*.pt`
- downloaded datasets
