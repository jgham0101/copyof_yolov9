# Week 5 - Baseline vs Proposed Comparison Pipeline

Comparison groups:
1. `baseline_yolov9_s_nms`: YOLOv9-S baseline with existing NMS validation.
2. `proposed_v10dual_nms`: YOLOv9-S + V10DualDDetect with NMS reference.
3. `proposed_v10dual_no_nms`: YOLOv9-S + V10DualDDetect with no-NMS top-k postprocess.

Default setting: COCO128, image size 320, 1 epoch.
This is a pipeline validation experiment, not the final paper-scale result.
