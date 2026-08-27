from __future__ import annotations
import argparse, json
from pathlib import Path
import torch


def tload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def unwrap(x):
    return (x.get("ema") or x.get("model")) if isinstance(x, dict) else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True)
    ap.add_argument("--dual", required=True)
    ap.add_argument("--out", required=True)
    a=ap.parse_args()
    c,d = unwrap(tload(a.control)).float(), unwrap(tload(a.dual)).float()
    cs,ds = c.state_dict(), d.state_dict()
    assert set(cs) == set(ds)
    hp = f"model.{len(c.model)-1}."
    groups = {"body":[], "o2m_head":[], "o2o_head":[]}
    for k in cs:
        if not k.startswith(hp): g="body"
        elif "one2one_" in k: g="o2o_head"
        else: g="o2m_head"
        diff=(cs[k].float()-ds[k].float()).abs()
        if diff.numel():
            groups[g].append((float(diff.max()), float(diff.mean())))
    report={}
    for g,rows in groups.items():
        report[g]={
            "tensor_count":len(rows),
            "max_abs_diff":max((x[0] for x in rows), default=0.0),
            "mean_of_tensor_mean_abs_diff":sum(x[1] for x in rows)/max(len(rows),1),
        }
    Path(a.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
