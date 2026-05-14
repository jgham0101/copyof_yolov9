from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from models.yolo import DetectionModel, V10DualDDetect

CFG = str(ROOT / "models/detect/yolov9-s-v10dual.yaml")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = DetectionModel(CFG, ch=3, nc=80).to(device)
head = model.model[-1]

assert isinstance(head, V10DualDDetect), f"Unexpected head type: {type(head)}"
assert head.nl == 3, f"Expected 3 detection layers, got {head.nl}"

model.train()
x = torch.zeros(1, 3, 256, 256, device=device)
out = model(x)

assert isinstance(out, dict), f"Training output must be dict, got {type(out)}"
assert set(out.keys()) == {"one2many", "one2one"}
assert len(out["one2many"]) == 3
assert len(out["one2one"]) == 3

for i, (a, b) in enumerate(zip(out["one2many"], out["one2one"])):
    assert a.shape == b.shape, f"Branch shape mismatch at P{i}: {a.shape} vs {b.shape}"
    assert a.shape[1] == head.no, f"Unexpected channels: {a.shape[1]} vs {head.no}"

dummy_loss = sum(t.float().mean() for t in out["one2many"]) + sum(t.float().mean() for t in out["one2one"])
dummy_loss.backward()

assert any(p.grad is not None for p in head.cv2.parameters()), "one2many head no grad"
assert any(p.grad is not None for p in head.one2one_cv2.parameters()), "one2one head no grad"

model.eval()
with torch.no_grad():
    pred = model(x)

assert isinstance(pred, tuple), f"Eval output must be tuple, got {type(pred)}"
assert isinstance(pred[0], list), f"Decoded output must be list, got {type(pred[0])}"
assert len(pred[0]) == 2, f"Expected two decoded branches, got {len(pred[0])}"
assert pred[0][0].ndim == 3 and pred[0][1].ndim == 3
assert isinstance(pred[1], dict)
assert "one2many" in pred[1] and "one2one" in pred[1]

print("Week 2 V10DualDDetect smoke test passed")
print("device:", device)
print("head:", type(head).__name__)
print("train one2many:", [tuple(t.shape) for t in out["one2many"]])
print("train one2one :", [tuple(t.shape) for t in out["one2one"]])
print("eval decoded:", [tuple(t.shape) for t in pred[0]])
