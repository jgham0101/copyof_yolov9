from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.yolo import V10DualActiveForwardDetect


def tload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def unwrap(ckpt):
    if isinstance(ckpt, dict):
        return ckpt.get("ema") or ckpt.get("model")
    return ckpt


def grad_sum(params):
    total = 0.0
    count = 0

    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().abs().sum().item())
            count += 1

    return total, count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    model = unwrap(tload(a.weights)).float().to(device)
    model.train()
    model.zero_grad(set_to_none=True)

    head = model.model[-1]
    assert isinstance(head, V10DualActiveForwardDetect)

    # Prevent BatchNorm running-stat changes from obscuring this gradient-only probe.
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    os.environ["YOLO_WEEK27_AUDIT_RETURN_O2O"] = "1"

    x = torch.rand(
        1, 3, 640, 640,
        device=device,
        requires_grad=True,
    )

    bundle = model(x)
    o2o_raw = bundle["o2o_raw"]

    # O2O-only scalar. No O2M tensor is included.
    scalar = sum(
        tensor.float().mean()
        for tensor in o2o_raw
    )

    scalar.backward()

    del os.environ["YOLO_WEEK27_AUDIT_RETURN_O2O"]

    o2o_params = (
        list(head.one2one_cv2.parameters())
        + list(head.one2one_cv3.parameters())
    )

    o2m_params = (
        list(head.cv2.parameters())
        + list(head.cv3.parameters())
    )

    # Upstream body excludes the final detection head.
    body_params = []
    for module in model.model[:-1]:
        body_params.extend(list(module.parameters()))

    o2o_grad, o2o_grad_tensors = grad_sum(o2o_params)
    o2m_grad, o2m_grad_tensors = grad_sum(o2m_params)
    body_grad, body_grad_tensors = grad_sum(body_params)

    input_grad = (
        float(x.grad.detach().abs().sum().item())
        if x.grad is not None
        else 0.0
    )

    report = {
        "probe_scalar": float(scalar.detach().item()),
        "o2o_head_grad_abs_sum": o2o_grad,
        "o2o_head_grad_tensor_count": o2o_grad_tensors,
        "o2m_head_grad_abs_sum": o2m_grad,
        "o2m_head_grad_tensor_count": o2m_grad_tensors,
        "body_grad_abs_sum": body_grad,
        "body_grad_tensor_count": body_grad_tensors,
        "input_grad_abs_sum": input_grad,
        "o2o_receives_gradient": bool(o2o_grad > 0.0),
        "o2m_isolated_from_o2o_scalar": bool(o2m_grad == 0.0),
        "body_isolated_from_o2o_scalar": bool(body_grad == 0.0),
        "input_isolated_from_o2o_scalar": bool(input_grad == 0.0),
    }

    report["detach_gradient_pass"] = bool(
        report["o2o_receives_gradient"]
        and report["o2m_isolated_from_o2o_scalar"]
        and report["body_isolated_from_o2o_scalar"]
        and report["input_isolated_from_o2o_scalar"]
    )

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
