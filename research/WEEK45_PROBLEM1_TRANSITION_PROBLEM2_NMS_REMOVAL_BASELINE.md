# Week45 — Problem 1 Transition + Problem 2 NMS-Removal Baseline

## Experiment status

- Training performed: `False`
- Evaluation already completed before this Git finalization.
- Problem 1 final status:
  `ANALYSIS_COMPLETED_WITH_ACCEPTED_RESIDUAL`
- Problem 1 strict closure gate: `FAIL`
- Strict gate relaxed: `False`
- Problem 2 progression:
  `ALLOWED_BY_RESEARCH_PROGRESSION_DECISION`

## Frozen references

- Baseline:
  Week36 native YOLOv9-S 20e
- Proposed Problem 2 source:
  Week39 O2O cls×2 20e
- Week44:
  negative ablation; not used as Problem 2 source

## Dataset contract

Canonical COCO2017 val2017:

- validation images: `5000`
- YOLO GT boxes: `36335`
- labeled images: `4952`
- background images: `48`
- classes: `80`

Dataset/checkpoint files are stored outside Git.

## Problem 2 isolation contract

The same decoded Week39 O2O tensor is passed to:

1. native NMS
2. YOLOv10-style no-NMS

Native NMS:

- confidence threshold: `0.001`
- NMS IoU: `0.7`
- max_det: `300`

No-NMS:

1. anchor max-class top-k
2. selected-anchor all-class flatten
3. global anchor × class top-k
4. confidence threshold
5. no NMS

## Week39 reproduction gate

- internal mAP50-95 reference: `0.429`
- COCO API AP50-95 reference: `0.428`
- tolerance: `±0.005`

Both internal and COCO API checks must pass before
interpreting the no-NMS accuracy delta.

## Problem 1 transition summary

# Problem 1 Final Summary / Transition to Problem 2

- Final status: **ANALYSIS_COMPLETED_WITH_ACCEPTED_RESIDUAL**
- Strict 98% closure gate: **FAIL (not relaxed)**
- Problem 2 progression: **ALLOWED_BY_RESEARCH_PROGRESSION_DECISION**
- Frozen baseline: Week36 — internal mAP50-95 0.453, COCO API AP50-95 0.454
- Frozen proposed source: Week39 cls×2 20e — internal mAP50-95 0.429, COCO API AP50-95 0.428
- Week44: negative ablation; preserved but not used as Problem 2 source.

Problem 1의 residual을 strict closure 성공으로 포장하지 않는다. Week45부터는 동일 Week39 O2O decoded output에서 native NMS와 YOLOv10-style no-NMS postprocess만 바꾸어 Problem 2의 추가 성능 하락과 latency 이득을 분리 측정한다.


---

## Week45 completed result summary

# Week45 Result Summary

## Reproduction gate
- Status: **INTERPRETATION_BLOCKED_REPRODUCTION_FAIL**
- Native +NMS internal mAP50-95: 0.454610 (Week39 ref 0.429, |Δ|=0.025610)
- Native +NMS COCO API AP50-95: 0.455892 (Week39 ref 0.428, |Δ|=0.027892)

## Problem 2 paired comparison
- Internal mAP50-95: +NMS 0.454610 → no-NMS 0.059173 (Δ=-0.395437)
- COCO API AP50-95: +NMS 0.455892 → no-NMS 0.097693 (Δ=-0.358200)
- Internal P: +NMS 0.680427 → no-NMS 0.098591
- Internal R: +NMS 0.569581 → no-NMS 0.398878
- Same-class duplicate pairs/image @ IoU≥0.7: +NMS 0.034800 → no-NMS 684.927600

## Latency (batch=1)
- Native NMS postprocess mean: 1.2049 ms
- no-NMS postprocess mean: 0.7373 ms
- Saved postprocess mean: 0.4677 ms
- Postprocess speedup ratio: 1.634×
- Pipeline total mean: native 28.6929 ms → no-NMS 28.2252 ms

## Week46 routing
- **REPAIR_WEEK39_REPRODUCTION_BEFORE_PROBLEM2_INTERPRETATION**
- Week39 +NMS reproduction gate failed; no-NMS causal interpretation is blocked.

## Interpretation rule
Week39 reproduction FAILED. Preserve these measurements, but do not claim the observed no-NMS delta is a valid Problem 2 causal result until reproduction is repaired.


---

## Git provenance

- Repository: `https://github.com/jgham0101/copyof_yolov9.git`
- Source branch used when Week45 evaluation started:
  `week39-one-factor-o2o-cls-scale-ablation`
- Experiment code base commit:
  `36954400c18497231a2224641fef2f1fcf89f995`
- Week45 branch:
  `week45_problem2_nms_removal_baseline`
- Git finalization performed **after** evaluation completion.
- This Git finalization does not recompute or modify metrics.
- Dataset, checkpoints and large prediction JSON files are
  intentionally not committed.
