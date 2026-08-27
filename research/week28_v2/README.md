# Week28 v2 — O2O Assignment/Loss Dual-Training Isolation

Base: week27-active-o2o-forward-equivalence

Week28 v1 is preserved but INVALID for scientific use.
Reason: literal backslash-n label generation corrupted 4520/5000 labels, leaving 480 images and 60 batches/epoch.

Week28 v2 hardening:
- persistent Drive static assets; local /content staging for training
- actual-newline label writer
- full 5000-image JPEG and 5000-label parsing gates
- no persistent dataset cache; fresh cache each session
- loader must be dataset=5000 and batches=625 at batch8
- same 5000/625 gate repeated inside train_week28_v2.py
- safe class-definition order
- explicit dual-training flag preserves stride bootstrap
- PyTorch 2.6 compatibility applied before checkpoint loads
- post-training strip/fuse/final-best validation disabled; external full-val used
- native YOLOv9 ComputeLoss reused directly for O2M top10 and O2O top1
- G4 CUDA tolerance fixed at 1e-5 from the start

Controlled factor:
- CONTROL: O2O loss weight 0
- DUAL: O2O loss weight 1

Still excluded: O2O inference, no-NMS, ranking/suppression auxiliaries, PSS/POTO/3DMF.
