from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.yolo import Model, V10DualActiveForwardDetect, V10DualTrainingDetectV2


def tload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def unwrap(ckpt):
    return (ckpt.get("ema") or ckpt.get("model")) if isinstance(ckpt, dict) else ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()

    src = unwrap(tload(a.source)).float().eval()
    assert isinstance(src.model[-1], V10DualActiveForwardDetect)
    tgt = Model(a.cfg, ch=3, nc=src.nc, anchors=None).float()
    assert isinstance(tgt.model[-1], V10DualTrainingDetectV2)

    s, t = src.state_dict(), tgt.state_dict()
    if set(s) != set(t):
        raise RuntimeError(f"state keys differ: src-only={sorted(set(s)-set(t))[:10]} tgt-only={sorted(set(t)-set(s))[:10]}")
    mapped = {}
    for k, tv in t.items():
        if s[k].shape != tv.shape:
            raise RuntimeError(f"shape mismatch {k}")
        mapped[k] = s[k].clone()
    tgt.load_state_dict(mapped, strict=True)
    tgt.names = copy.deepcopy(src.names)
    tgt.nc = src.nc

    loaded = tgt.state_dict()
    max_diff = max(
        float((loaded[k].float()-v.float()).abs().max().item()) if v.numel() else 0.0
        for k,v in mapped.items()
    )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": -1, "best_fitness": None,
        "model": copy.deepcopy(tgt).half(), "ema": None, "updates": None,
        "optimizer": None,
        "week28v2_note": "Exact Week27 state; explicit dual-training return/loss only",
    }, out)
    report = {
        "source_state_items": len(s), "target_state_items": len(t),
        "mapped_state_items": len(mapped), "postload_max_abs_diff": max_diff,
        "target_stride": [int(x) for x in tgt.model[-1].stride.tolist()],
    }
    Path(a.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
