# Week 2 v2 - V10DualDDetect Clean Integration

Main files:
- `models/yolo.py`
- `utils/loss_tal_dual.py`
- `models/detect/yolov9-s-v10dual.yaml`
- `research/week2/check_v10dual_head.py`

Design:
- one2many branch: dense supervision
- one2one branch: detached feature and top-k=1 assigner
- validation compatibility: returns `[one2many_decoded, one2one_decoded]`
