# Week 3 - No-NMS Inference Path

## Goal

Implement an inference path that uses the `one2one` branch from `V10DualDDetect` and removes NMS from post-processing.

## Added files

- `utils/e2e_postprocess.py`
- `detect_e2e.py`
- `research/week3/check_e2e_postprocess.py`
- `research/week3/benchmark_postprocess.py`

## Postprocess design

`v10_no_nms_postprocess()`:
1. Selects `one2one` decoded prediction.
2. Takes the best class score per prediction.
3. Filters by confidence.
4. Applies global top-k.
5. Converts boxes from xywh to xyxy.
6. Returns `[x1, y1, x2, y2, conf, cls]`.

This is intentionally different from NMS: it performs no IoU-based suppression.

## Week 3 scope

This week verifies path correctness and latency of the postprocess stage.
Full COCO-style mAP evaluation with no-NMS should be implemented in Week 4.
