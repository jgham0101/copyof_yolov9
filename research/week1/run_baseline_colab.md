# Week 1 Colab Baseline Commands

## Train smoke test

```bash
python train_dual.py \
  --workers 2 \
  --device 0 \
  --batch 8 \
  --data data/coco128.yaml \
  --img 640 \
  --cfg models/detect/yolov9-s.yaml \
  --weights "" \
  --name week1_yolov9_baseline_coco128_smoke \
  --hyp hyp.scratch-high.yaml \
  --epochs 1 \
  --close-mosaic 0
```

If Colab memory is insufficient:
- set `--batch 4`
- set `--img 512`

## Validation

```bash
python val_dual.py \
  --data data/coco128.yaml \
  --img 640 \
  --batch 8 \
  --weights runs/train/week1_yolov9_baseline_coco128_smoke/weights/last.pt \
  --device 0 \
  --name week1_yolov9_baseline_val
```
