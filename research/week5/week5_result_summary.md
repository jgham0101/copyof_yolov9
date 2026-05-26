# Week 5 Result Summary - Baseline vs Proposed Comparison

## Branch
- `week5-experiment-comparison`

## Scope
This week builds a comparison pipeline under the same small-scale COCO128 condition.

## Compared groups
1. baseline_yolov9_s + NMS
2. proposed_v10dual + NMS
3. proposed_v10dual + no-NMS

## Dataset and training
- Dataset: COCO128
- Image size: 320
- Epochs: 1
- Batch size: 4
- Environment: Colab free-tier compatible

## Outputs
- `/content/drive/MyDrive/yolo_v9_v10_e2e_research/week5_experiment_comparison/week5_experiment_comparison.csv`
- `/content/drive/MyDrive/yolo_v9_v10_e2e_research/week5_experiment_comparison/week5_experiment_comparison.json`
- Baseline validation log
- Proposed NMS/no-NMS metrics.json files

## Interpretation
This is not the final paper-scale result.
The goal is to prove that baseline/proposed training and evaluation can be run under a controlled comparison pipeline.
Final quantitative claims require larger dataset and longer training.

## Next
Week 6 should run longer experiments and prepare paper tables.