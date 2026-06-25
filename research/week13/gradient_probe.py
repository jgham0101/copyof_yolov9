
import argparse
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.yolo import Model

def flatten_tensors(obj):
    out = []
    if torch.is_tensor(obj):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(flatten_tensors(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(flatten_tensors(v))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--nc", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model(args.cfg, ch=3, nc=args.nc).to(device)
    model.train()

    x = torch.randn(2, 3, args.imgsz, args.imgsz, device=device)
    y = model(x)
    tensors = [t for t in flatten_tensors(y) if torch.is_tensor(t) and torch.is_floating_point(t)]
    if not tensors:
        raise RuntimeError("no float tensor outputs")

    loss = sum(t.float().mean() for t in tensors)
    model.zero_grad(set_to_none=True)
    loss.backward()

    modules = []
    total_params = 0
    params_with_grad = 0

    for name, module in model.named_modules():
        cls = module.__class__.__name__
        if "Detect" in cls or "DDetect" in cls or "V10Dual" in cls:
            n_params = 0
            n_grad = 0
            grad_abs_sum = 0.0
            grad_abs_max = 0.0
            for p in module.parameters(recurse=True):
                n_params += p.numel()
                if p.grad is not None:
                    n_grad += p.numel()
                    g = p.grad.detach().abs()
                    grad_abs_sum += float(g.sum().item())
                    grad_abs_max = max(grad_abs_max, float(g.max().item()))
            modules.append({
                "module_name": name,
                "class": cls,
                "num_params": n_params,
                "num_params_with_grad": n_grad,
                "grad_abs_sum": grad_abs_sum,
                "grad_abs_max": grad_abs_max,
            })

    for p in model.parameters():
        total_params += p.numel()
        if p.grad is not None:
            params_with_grad += p.numel()

    result = {
        "cfg": args.cfg,
        "nc": args.nc,
        "imgsz": args.imgsz,
        "dummy_loss": float(loss.detach().item()),
        "num_output_tensors": len(tensors),
        "used_output_shapes": [list(t.shape) for t in tensors[:50]],
        "total_params": total_params,
        "params_with_grad": params_with_grad,
        "detect_modules": modules,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
