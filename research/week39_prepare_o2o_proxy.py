
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import torch

def tload(path):
    try:
        return torch.load(path,map_location="cpu",weights_only=False)
    except TypeError:
        return torch.load(path,map_location="cpu")

def max_module_diff(a,b):
    sa=a.state_dict();sb=b.state_dict()
    if set(sa)!=set(sb):raise RuntimeError("state-key mismatch")
    m=0.0
    for k in sa:
        x=sa[k].float();y=sb[k].float()
        if x.shape!=y.shape:raise RuntimeError(k)
        if x.numel():m=max(m,float((x-y).abs().max()))
    return m

def swap(model):
    if model is None:return None
    h=model.model[-1]
    for attr in ["cv2","cv3","one2one_cv2","one2one_cv3"]:
        if not hasattr(h,attr):raise RuntimeError(f"missing {attr}")
    h.cv2.load_state_dict(copy.deepcopy(h.one2one_cv2.state_dict()),strict=True)
    h.cv3.load_state_dict(copy.deepcopy(h.one2one_cv3.state_dict()),strict=True)
    bd=max_module_diff(h.cv2,h.one2one_cv2)
    cd=max_module_diff(h.cv3,h.one2one_cv3)
    if bd!=0.0 or cd!=0.0:raise RuntimeError((bd,cd))
    if hasattr(h,"week28_dual_training"):h.week28_dual_training=False
    return {"box_postcopy_max_abs_diff":bd,"cls_postcopy_max_abs_diff":cd}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--report",required=True)
    a=ap.parse_args()
    ck=tload(a.source)
    if not isinstance(ck,dict):raise TypeError(type(ck))
    out=copy.deepcopy(ck)
    rep={}
    for key in ["model","ema"]:
        rep[key]=swap(out.get(key)) if out.get(key) is not None else None
    out["week39_inference_mode"]="O2O bbox/cls copied into native inference slots."
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    torch.save(out,a.out)
    report={"source":a.source,"output":a.out,"modules":rep,"pass":True}
    Path(a.report).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
