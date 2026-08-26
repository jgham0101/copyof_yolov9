from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.yolo import Model, V10DualDormantDetect, V10DualActiveForwardDetect


def tload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def unwrap(ckpt):
    if isinstance(ckpt, dict):
        return ckpt.get("ema") or ckpt.get("model")
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()

    src = unwrap(tload(a.source)).float().eval()
    assert isinstance(src.model[-1], V10DualDormantDetect)

    tgt = Model(a.cfg, ch=3, nc=src.nc, anchors=None).float()
    assert isinstance(tgt.model[-1], V10DualActiveForwardDetect)

    src_sd = src.state_dict()
    tgt_sd = tgt.state_dict()

    if set(src_sd.keys()) != set(tgt_sd.keys()):
        only_src = sorted(set(src_sd) - set(tgt_sd))
        only_tgt = sorted(set(tgt_sd) - set(src_sd))
        raise RuntimeError(
            "Week26/27 state key mismatch\\n"
            f"only_src={only_src[:20]}\\n"
            f"only_tgt={only_tgt[:20]}"
        )

    mapped = {}

    for k, tv in tgt_sd.items():
        sv = src_sd[k]

        if sv.shape != tv.shape:
            raise RuntimeError(
                f"Shape mismatch {k}: {sv.shape} vs {tv.shape}"
            )

        mapped[k] = sv.clone()

    tgt.load_state_dict(mapped, strict=True)
    tgt.names = copy.deepcopy(src.names)
    tgt.nc = src.nc

    loaded = tgt.state_dict()
    max_diff = 0.0
    exact = 0

    for k, v in mapped.items():
        diff = (loaded[k].float() - v.float()).abs()
        md = float(diff.max().item()) if diff.numel() else 0.0
        max_diff = max(max_diff, md)
        exact += int(torch.equal(loaded[k], v))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": -1,
            "best_fitness": None,
            "model": copy.deepcopy(tgt).half(),
            "ema": None,
            "updates": None,
            "optimizer": None,
            "week27_note": (
                "Week26 exact state; only detached O2O forward activated. "
                "O2O loss/inference remain disabled."
            ),
        },
        out,
    )

    report = {
        "source_head": type(src.model[-1]).__name__,
        "target_head": type(tgt.model[-1]).__name__,
        "source_state_items": len(src_sd),
        "target_state_items": len(tgt_sd),
        "mapped_state_items": len(mapped),
        "unmapped_target_items": 0,
        "postload_max_abs_diff": max_diff,
        "postload_exact_tensor_count": exact,
    }

    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
