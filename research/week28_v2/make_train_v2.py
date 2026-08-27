from pathlib import Path

src = Path("train.py")
dst = Path("train_week28_v2.py")
text = src.read_text(encoding="utf-8")

old_import = "from utils.loss_tal import ComputeLoss"
if old_import not in text:
    raise RuntimeError("Native loss import not found")
text = text.replace(
    old_import,
    "from utils.loss_tal_week28_v2 import ComputeLossWeek28V2 as ComputeLoss",
    1,
)

# PyTorch 2.6 compatibility in the standalone copied train script.
text = text.replace(
    "torch.load(weights, map_location='cpu')",
    "torch.load(weights, map_location='cpu', weights_only=False)",
)

# Explicitly activate dual-training return only after Model construction and attributes.
anchor_names = "    model.names = names\n"
if anchor_names not in text:
    raise RuntimeError("model.names anchor missing")
activation = (
    anchor_names
    + "    week28_head = de_parallel(model).model[-1]\n"
    + "    if not hasattr(week28_head, 'week28_dual_training'):\n"
    + "        raise RuntimeError('Week28-v2 dual-training flag missing')\n"
    + "    week28_head.week28_dual_training = True\n"
)
text = text.replace(anchor_names, activation, 1)

# Dataset-size gate immediately after dataset construction.
anchor_labels = "    labels = np.concatenate(dataset.labels, 0)\n"
if anchor_labels not in text:
    raise RuntimeError("dataset labels anchor missing")
dataset_gate = (
    "    expected_images = int(os.getenv('YOLO_WEEK28_V2_EXPECTED_IMAGES', '-1'))\n"
    "    if expected_images > 0 and len(dataset) != expected_images:\n"
    "        raise RuntimeError(f'Week28-v2 dataset mismatch: {len(dataset)} != {expected_images}')\n"
    + anchor_labels
)
text = text.replace(anchor_labels, dataset_gate, 1)

# Batch-count gate.
anchor_nb = "    nb = len(train_loader)  # number of batches\n"
if anchor_nb not in text:
    raise RuntimeError("nb anchor missing")
nb_gate = (
    anchor_nb
    + "    expected_batches = int(os.getenv('YOLO_WEEK28_V2_EXPECTED_BATCHES', '-1'))\n"
    + "    if expected_batches > 0 and nb != expected_batches:\n"
    + "        raise RuntimeError(f'Week28-v2 batch mismatch: {nb} != {expected_batches}')\n"
    + "    LOGGER.info(f'Week28V2 DATA GATE PASS: images={len(dataset)} batches={nb}')\n"
)
text = text.replace(anchor_nb, nb_gate, 1)

# Preserve official global clipping; record pre-clip norm returned by clip_grad_norm_.
old_clip = "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients"
if old_clip not in text:
    raise RuntimeError("clip_grad_norm anchor missing")
new_clip = (
    "week28_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients\n"
    "                compute_loss.record_clip_norm(float(week28_grad_norm), ni, epoch)"
)
text = text.replace(old_clip, new_clip, 1)

# External full validation only. --noval means no internal val even at final epoch.
text = text.replace(
    "if not noval or final_epoch:  # Calculate mAP",
    "if not noval:  # Week28 v2: external full-val only",
    1,
)
text = text.replace("compute_loss=compute_loss)", "compute_loss=None)")

# Skip post-training strip/fuse/final-best validation completely.
post = "        for f in last, best:\n"
if post not in text:
    raise RuntimeError("post-training loop anchor missing")
text = text.replace(post, "        for f in ():  # Week28 v2: skip strip/fuse/final validation\n", 1)

dst.write_text(text, encoding="utf-8")
print("created:", dst)
