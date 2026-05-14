# Week 1 Result Summary - User Fork Baseline

## Source repositories
- YOLOv9 baseline fork: https://github.com/jgham0101/copyof_yolov9
- YOLOv10 reference fork: https://github.com/jgham0101/copyof_yolov10

## Commit snapshot
- copyof_yolov9: `5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff`
- copyof_yolov10: `453c6e38a51e9d1d5a2aa5fb7f1014a711913397`

## Week 1 scope
- No model architecture modification yet.
- Verify baseline execution.
- Identify YOLOv9 modification points.
- Identify YOLOv10 reference points for one2many / one2one head design.

## Generated files
- `research/week1/README.md`
- `research/week1/analyze_code_structure.py`
- `research/week1/run_baseline_colab.md`
- `research/week1/yolov9_code_structure.csv`
- `research/week1/yolov10_reference_structure.csv`

## Drive backup
- `/content/drive/MyDrive/yolo_v9_v10_e2e_research/week1_user_fork`

## Baseline run
- Train: `runs/train/week1_yolov9_baseline_coco128_smoke`
- Validation: `runs/val/week1_yolov9_baseline_val2`

## Week 2 entry condition
Proceed to Week 2 only if:
1. `train_dual.py` finishes at least 1 epoch.
2. `val_dual.py` runs with the generated weight.
3. `research/week1/yolov9_code_structure.csv` contains Head, Loss, NMS locations.
4. `research/week1/yolov10_reference_structure.csv` contains one2one/one2many locations.
