# Week 6 Result Summary - Paper Analysis Preparation

## Scope

Week 6 organizes the Week 5 comparison results into paper-style tables and optionally repeats V10DualDDetect NMS/no-NMS validation to inspect latency stability.

## Core idea

The research is not yet claiming final accuracy improvement. The goal of Week 6 is to prepare reliable result tables and interpretation notes from the existing smoke-scale pipeline.

## Generated outputs

- `accuracy_table.csv`
- `latency_table.csv`
- `postprocess_comparison.csv`
- `week6_paper_tables.md`
- `paper_tables_summary.json`
- optional `week6_repeat_eval_results.csv`

## Interpretation

The current default result is based on COCO128, image size 320, and short training. Therefore, mAP values are not final paper-scale performance values.

The current paper-level claim should be limited to:

> A complete baseline/proposed comparison pipeline was built, including V10DualDDetect NMS and no-NMS evaluation paths, and the results can be summarized into accuracy and latency tables.

## Next step

Run longer training or a larger subset dataset and regenerate the same Week 6 tables.
