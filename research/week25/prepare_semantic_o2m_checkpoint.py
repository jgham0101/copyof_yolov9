from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.yolo import Model, DualDDetect, V10O2MDetect

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
    assert isinstance(src.model[-1], DualDDetect)

    tgt = Model(a.cfg, ch=3, nc=src.nc, anchors=None).float()
    assert isinstance(tgt.model[-1], V10O2MDetect)

    src_sd = src.state_dict()
    tgt_sd = tgt.state_dict()

    si = len(src.model) - 1
    ti = len(tgt.model) - 1
    sp = f"model.{si}."
    tp = f"model.{ti}."

    mapped = {}
    semantic = {}

    for tk, tv in tgt_sd.items():
        if tk.startswith(tp):
            continue
        if tk not in src_sd:
            raise RuntimeError(f"Missing body source key: {tk}")
        if src_sd[tk].shape != tv.shape:
            raise RuntimeError(
                f"Body shape mismatch: {tk} {src_sd[tk].shape} -> {tv.shape}"
            )
        mapped[tk] = src_sd[tk].clone()

    for src_stem, tgt_stem in [
        ("cv4.", "cv2."),
        ("cv5.", "cv3."),
        ("dfl2.", "dfl."),
    ]:
        src_base = sp + src_stem
        tgt_base = tp + tgt_stem
        src_keys = [k for k in src_sd if k.startswith(src_base)]
        if not src_keys:
            raise RuntimeError(f"No keys under {src_base}")

        for sk in src_keys:
            suffix = sk[len(src_base):]
            tk = tgt_base + suffix
            if tk not in tgt_sd:
                raise RuntimeError(f"Missing semantic target: {sk} -> {tk}")
            if src_sd[sk].shape != tgt_sd[tk].shape:
                raise RuntimeError(
                    f"Semantic shape mismatch: {sk} -> {tk}: "
                    f"{src_sd[sk].shape} vs {tgt_sd[tk].shape}"
                )
            mapped[tk] = src_sd[sk].clone()
            semantic[tk] = sk

    missing = [k for k in tgt_sd if k not in mapped]
    if missing:
        raise RuntimeError(
            "Unmapped target keys:\n" + json.dumps(missing[:50], indent=2)
        )

    tgt.load_state_dict(mapped, strict=True)
    tgt.names = copy.deepcopy(src.names)
    tgt.nc = src.nc

    loaded = tgt.state_dict()
    max_diff = 0.0
    exact = 0
    for k, v in mapped.items():
        d = (loaded[k].float() - v.float()).abs()
        md = float(d.max().item()) if d.numel() else 0.0
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
            "week25_note": (
                "official YOLOv9 body + semantic main head "
                "cv4/cv5/dfl2 -> cv2/cv3/dfl"
            ),
        },
        out,
    )

    report = {
        "source_head_class": type(src.model[-1]).__name__,
        "target_head_class": type(tgt.model[-1]).__name__,
        "source_head_index": si,
        "target_head_index": ti,
        "target_state_items": len(tgt_sd),
        "mapped_state_items": len(mapped),
        "semantic_head_items": len(semantic),
        "unmapped_target_items": len(missing),
        "postload_max_abs_diff": max_diff,
        "postload_exact_tensor_count": exact,
        "semantic_mapping": semantic,
    }

    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(
        {k: v for k, v in report.items() if k != "semantic_mapping"},
        indent=2
    ))

if __name__ == "__main__":
    main()
