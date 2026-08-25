# Week25 — Native YOLOv9 -> V10 O2M Functional Equivalence Audit

This branch starts from official YOLOv9 upstream/main.
Week24 and earlier research branches remain preserved.

Week25 intentionally contains:
- no training
- no O2O
- no no-NMS
- no ranking auxiliary loss
- no competitor suppression

Goal:
Preserve official YOLOv9-S main-branch detection capability while replacing
the native DualDDetect main inference path with a single V10O2MDetect scaffold.

Semantic mapping:
- cv4 -> cv2
- cv5 -> cv3
- dfl2 -> dfl
- feature inputs [15,18,21]

Week26 is allowed only after all Week25 equivalence gates pass.
