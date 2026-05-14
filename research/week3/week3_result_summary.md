# Week 3 Result Summary - No-NMS Inference Path

## Branch
- `week3-nms-free-inference`

## Implemented
- `utils/e2e_postprocess.py`
- `detect_e2e.py`
- `research/week3/check_e2e_postprocess.py`
- `research/week3/benchmark_postprocess.py`

## Theory
- Uses the one-to-one branch for inference.
- Removes IoU-based suppression.
- Applies confidence filtering and top-k selection only.
- Keeps NMS reference path only for latency comparison.

## Drive backup
- `/content/drive/MyDrive/yolo_v9_v10_e2e_research/week3_nms_free_inference`

## Week 4 target
- Build `val_e2e.py` for COCO-style AP evaluation without NMS.
- Compare AP / latency / FPS between NMS and no-NMS paths.
