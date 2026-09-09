# Week43 — AP-Critical Failure Subset Tracing

- Total val GT: 36335
- Baseline same-class IoU>=0.50 hits: 32036
- Proposed same-class IoU>=0.50 hits: 31581
- AP-critical GT: 849 (2.34%)

## Ranked failure types
1. CORRECT_CLASS_CONFIDENCE_FAILURE: 491 (57.83%)
2. NMS_OR_MAXDET_FAILURE: 110 (12.96%)
3. ANCHOR_SELECTION_GAP: 109 (12.84%)
4. RAW_LOCALIZATION_FAILURE: 78 (9.19%)
5. SAME_ANCHOR_HEAD_GAP: 25 (2.94%)
6. NO_O2O_POSITIVE_ASSIGNMENT: 19 (2.24%)
7. MIXED_HEAD_SELECTION: 17 (2.00%)

- Dominant: **CORRECT_CLASS_CONFIDENCE_FAILURE** (57.83%)
- Top2 cumulative: 70.79%
- Concentration: **SINGLE_DOMINANT_FAILURE**

- Problem 1: **RESIDUAL_REMAINS**
- Problem 2: **BLOCKED**
- Week44: **WEEK44_ONE_FACTOR_CRITICAL_CORRECT_CLASS_CONFIDENCE_REFINEMENT**

AP-critical GT는 full COCO AP matching 자체가 아니라 recall-side diagnostic proxy다.

## Provenance
- Base branch: `week40-target-quality-correct-class-confidence-diagnostic`
- Base commit: `96457455fdf931aa59846eea391a7416ff8f4335`
- Baseline SHA256: `c50310277af9c7eb8f951fe86787c15d793076ef7f60be853927a827e6ce2a61`
- Proposed SHA256: `f766c69e067981fe24beed0851637710a28e6797b5dd4d5818753faf3eb41586`
- Training/backward/source modification: NO
