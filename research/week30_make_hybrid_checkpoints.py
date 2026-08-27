from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import torch

def tload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def module_maxdiff(a, b):
    sa = a.state_dict()
    sb = b.state_dict()
    assert set(sa) == set(sb)
    m = 0.0
    for k in sa:
        x = sa[k].float()
        y = sb[k].float()
        assert x.shape == y.shape
        if x.numel():
            m = max(m, float((x-y).abs().max().item()))
    return m

def apply_variant(model, box_src, cls_src):
    h = model.model[-1]
    for attr in ["cv2", "cv3", "one2one_cv2", "one2one_cv3"]:
        if not hasattr(h, attr):
            raise RuntimeError(f"head missing {attr}")

    if box_src == "O":
        h.cv2.load_state_dict(
            copy.deepcopy(h.one2one_cv2.state_dict()), strict=True
        )
    if cls_src == "O":
        h.cv3.load_state_dict(
            copy.deepcopy(h.one2one_cv3.state_dict()), strict=True
        )

    return {
        "box_source": box_src,
        "cls_source": cls_src,
        "box_postcopy_max_abs_diff": (
            module_maxdiff(h.cv2, h.one2one_cv2) if box_src == "O" else None
        ),
        "cls_postcopy_max_abs_diff": (
            module_maxdiff(h.cv3, h.one2one_cv3) if cls_src == "O" else None
        ),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()

    src = tload(a.source)
    assert isinstance(src, dict)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "M_M": ("M", "M"),
        "M_O": ("M", "O"),
        "O_M": ("O", "M"),
        "O_O": ("O", "O"),
    }

    report = {}

    for tag, (box_src, cls_src) in variants.items():
        ckpt = copy.deepcopy(src)
        item = {}

        for key in ["model", "ema"]:
            if ckpt.get(key) is not None:
                item[key] = apply_variant(
                    ckpt[key], box_src, cls_src
                )
            else:
                item[key] = None

        ckpt["week30_variant"] = {
            "tag": tag,
            "box_source": box_src,
            "cls_source": cls_src,
        }

        out = out_dir / f"week30_{tag}.pt"
        torch.save(ckpt, out)

        item["path"] = str(out)
        item["box_source"] = box_src
        item["cls_source"] = cls_src

        report[tag] = item

    Path(a.report).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
