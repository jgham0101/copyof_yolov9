
from __future__ import annotations

from pathlib import Path
import argparse
import inspect
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.experimental import attempt_load
import models.yolo as yolo


def tensor_summary(x):
    if not torch.is_tensor(x):
        return None
    d = {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
    }
    if x.numel():
        xf = x.detach().float()
        d.update({
            "min": float(xf.min().item()),
            "max": float(xf.max().item()),
            "mean": float(xf.mean().item()),
        })
    return d


def walk(obj, path="root", max_depth=5):
    items = []
    if max_depth < 0:
        return items

    if torch.is_tensor(obj):
        items.append({"path": path, "type": "Tensor", **tensor_summary(obj)})
    elif isinstance(obj, dict):
        items.append({"path": path, "type": "dict", "keys": list(obj.keys())})
        for k, v in obj.items():
            items.extend(walk(v, f"{path}.{k}", max_depth - 1))
    elif isinstance(obj, (list, tuple)):
        items.append({"path": path, "type": type(obj).__name__, "len": len(obj)})
        for i, v in enumerate(obj):
            items.extend(walk(v, f"{path}[{i}]", max_depth - 1))
    else:
        items.append({"path": path, "type": type(obj).__name__, "repr": repr(obj)[:300]})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = attempt_load(args.weights, device=device)
    model.eval()

    x = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
    with torch.no_grad():
        out = model(x)

    structure = walk(out)

    source = {}
    if hasattr(yolo, "V10DualDDetect"):
        cls = yolo.V10DualDDetect
        source["class_name"] = "V10DualDDetect"
        try:
            source["class_source"] = inspect.getsource(cls)
        except Exception as e:
            source["class_source_error"] = repr(e)
        try:
            source["forward_source"] = inspect.getsource(cls.forward)
        except Exception as e:
            source["forward_source_error"] = repr(e)

    modules = []
    for name, m in model.named_modules():
        if m.__class__.__name__ in ["V10DualDDetect", "DDetect", "DualDDetect"]:
            modules.append({
                "name": name,
                "class": m.__class__.__name__,
                "attrs": sorted([a for a in dir(m) if "one" in a.lower() or "many" in a.lower() or "cv" in a.lower()])[:100],
            })

    result = {
        "weights": args.weights,
        "imgsz": args.imgsz,
        "output_structure": structure,
        "detect_modules": modules,
        "source": source,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps({
        "weights": args.weights,
        "num_output_items": len(structure),
        "detect_modules": modules,
        "out": str(out_path),
    }, indent=2))


if __name__ == "__main__":
    main()
