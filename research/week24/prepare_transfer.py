
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path
import torch, yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.yolo import Model
from utils.general import intersect_dicts

def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def get_model(c):
    return (c.get("ema") or c.get("model")) if isinstance(c, dict) else c

def copy_head(model):
    h = model.model[-1]
    report = {"head_class": h.__class__.__name__, "copied": [], "errors": []}
    for sname, dname in [("cv2","one2one_cv2"),("cv3","one2one_cv3"),("dfl","dfl2")]:
        if not hasattr(h,sname) or not hasattr(h,dname):
            report["errors"].append(f"missing {sname}/{dname}"); continue
        s, d = getattr(h,sname), getattr(h,dname)
        try:
            if isinstance(s, torch.nn.ModuleList):
                assert isinstance(d, torch.nn.ModuleList) and len(s)==len(d)
                for i,(sm,dm) in enumerate(zip(s,d)):
                    dm.load_state_dict(copy.deepcopy(sm.state_dict()), strict=True)
                    report["copied"].append(f"{sname}[{i}]->{dname}[{i}]")
            else:
                d.load_state_dict(copy.deepcopy(s.state_dict()), strict=True)
                report["copied"].append(f"{sname}->{dname}")
        except Exception as e:
            report["errors"].append(repr(e))
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cfg",required=True); ap.add_argument("--data",required=True)
    ap.add_argument("--source",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--report",required=True); a=ap.parse_args()

    data=yaml.safe_load(Path(a.data).read_text())
    model=Model(a.cfg,ch=3,nc=len(data["names"]),anchors=None).float()
    src=get_model(load(a.source)).float()
    compatible=intersect_dicts(src.state_dict(),model.state_dict(),exclude=[])
    model.load_state_dict(compatible,strict=False)
    branch=copy_head(model)

    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "epoch":-1,"best_fitness":None,
        "model":copy.deepcopy(model).half(),"ema":None,
        "updates":None,"optimizer":None,
        "week24_note":"official transfer then O2M->O2O copy"
    },out)

    rep={
        "source":a.source,"out":str(out),
        "transferred_items":len(compatible),
        "target_items":len(model.state_dict()),
        "branch_copy":branch,
        "size":out.stat().st_size,
    }
    Path(a.report).write_text(json.dumps(rep,indent=2))
    print(json.dumps(rep,indent=2))

if __name__=="__main__": main()
