from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.yolo import DualDDetect, V10O2MDetect
from utils.general import non_max_suppression

def tload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def unwrap(ckpt):
    if isinstance(ckpt, dict):
        return ckpt.get("ema") or ckpt.get("model")
    return ckpt

def compare(name, a, b):
    if a.shape != b.shape:
        return {
            "name": name,
            "shape_a": str(list(a.shape)),
            "shape_b": str(list(b.shape)),
            "shape_equal": False,
            "max_abs_diff": float("inf"),
            "mean_abs_diff": float("inf"),
            "allclose": False,
        }

    af = a.float()
    bf = b.float()
    diff = (af - bf).abs()

    return {
        "name": name,
        "shape_a": str(list(a.shape)),
        "shape_b": str(list(b.shape)),
        "shape_equal": True,
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "allclose": bool(torch.allclose(af, bf, rtol=1e-5, atol=1e-6)),
    }

def register_hooks(model, store):
    handles = []
    for idx in [15, 18, 21]:
        def hook(_m, _inp, out, idx=idx):
            store[idx] = out.detach().clone()
        handles.append(model.model[idx].register_forward_hook(hook))
    return handles

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    native = unwrap(tload(a.native)).float().to(device).eval()
    target = unwrap(tload(a.target)).float().to(device).eval()

    assert isinstance(native.model[-1], DualDDetect)
    assert isinstance(target.model[-1], V10O2MDetect)

    torch.manual_seed(2025)
    torch.cuda.manual_seed_all(2025)
    x = torch.rand(1, 3, 640, 640, device=device)

    nf, tf = {}, {}
    hn = register_hooks(native, nf)
    ht = register_hooks(target, tf)

    with torch.no_grad():
        no = native(x)
        to = target(x)

    for h in hn + ht:
        h.remove()

    native_dec = no[0][1]
    native_raw = no[1][1]
    target_dec = to[0]
    target_raw = to[1]

    feature_df = pd.DataFrame([
        compare(f"layer_{i}", nf[i], tf[i])
        for i in [15, 18, 21]
    ])
    feature_df.to_csv(out / "week25_feature_equivalence.csv", index=False)

    raw_df = pd.DataFrame([
        compare(f"P{i+3}_raw", na, ta)
        for i, (na, ta) in enumerate(zip(native_raw, target_raw))
    ])
    raw_df.to_csv(out / "week25_raw_head_equivalence.csv", index=False)

    dec = compare("decoded", native_dec, target_dec)
    (out / "week25_decode_equivalence.json").write_text(
        json.dumps(dec, indent=2), encoding="utf-8"
    )

    native_nms = non_max_suppression(
        native_dec.clone(),
        conf_thres=0.001,
        iou_thres=0.7,
        multi_label=True,
        max_det=300,
    )
    target_nms = non_max_suppression(
        target_dec.clone(),
        conf_thres=0.001,
        iou_thres=0.7,
        multi_label=True,
        max_det=300,
    )

    nms_rows = []
    for i, (na, ta) in enumerate(zip(native_nms, target_nms)):
        row = {
            "batch_index": i,
            "native_count": int(na.shape[0]),
            "target_count": int(ta.shape[0]),
            "count_equal": int(na.shape[0]) == int(ta.shape[0]),
            "shape_equal": na.shape == ta.shape,
            "max_abs_diff": float("inf"),
            "allclose": False,
        }
        if na.shape == ta.shape:
            diff = (na.float() - ta.float()).abs()
            row["max_abs_diff"] = float(diff.max().item()) if diff.numel() else 0.0
            row["allclose"] = bool(
                torch.allclose(na.float(), ta.float(), rtol=1e-5, atol=1e-6)
            )
        nms_rows.append(row)

    nms_df = pd.DataFrame(nms_rows)
    nms_df.to_csv(out / "week25_nms_equivalence.csv", index=False)

    gate = {
        "feature_pass": bool(feature_df["allclose"].all()),
        "raw_pass": bool(raw_df["allclose"].all()),
        "decode_pass": bool(dec["allclose"]),
        "nms_pass": bool(nms_df["allclose"].all()),
        "feature_max_abs_diff": float(feature_df["max_abs_diff"].max()),
        "raw_max_abs_diff": float(raw_df["max_abs_diff"].max()),
        "decode_max_abs_diff": float(dec["max_abs_diff"]),
        "nms_max_abs_diff": float(nms_df["max_abs_diff"].max()),
    }

    (out / "week25_tensor_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )

    print("FEATURE")
    print(feature_df.to_string(index=False))
    print("\nRAW")
    print(raw_df.to_string(index=False))
    print("\nDECODE")
    print(json.dumps(dec, indent=2))
    print("\nNMS")
    print(nms_df.to_string(index=False))
    print("\nGATE")
    print(json.dumps(gate, indent=2))

if __name__ == "__main__":
    main()
