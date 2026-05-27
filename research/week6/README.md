# Week 6 - Paper Table and Repeated Evaluation

## Purpose

Week 6 organizes the Week 5 comparison results into paper-style tables and repeats the proposed model evaluation to inspect latency stability.

## Main outputs

- `accuracy_table.csv`
- `latency_table.csv`
- `postprocess_comparison.csv`
- `week6_paper_tables.md`
- `week6_repeat_eval_summary.csv`
- `week6_result_summary.md`

## Scope

The current experiment is COCO128 / 1 epoch / img 320, so it is not a final paper-scale result.

This week focuses on:
1. verifying that the full result pipeline is reproducible,
2. preparing paper table formats,
3. checking NMS vs no-NMS postprocess latency consistency.
