
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
import pandas as pd

def latest(repo,name):
    xs=sorted((repo/"runs/val").glob(name+"*/metrics.json"))
    if not xs: raise FileNotFoundError(name)
    return xs[-1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo",required=True); ap.add_argument("--weights",required=True)
    ap.add_argument("--data",required=True); ap.add_argument("--tag",required=True)
    ap.add_argument("--out",required=True); ap.add_argument("--imgsz",type=int,default=640)
    ap.add_argument("--batch",type=int,default=8); ap.add_argument("--conf",type=float,default=.001)
    ap.add_argument("--iou",type=float,default=.7); ap.add_argument("--max-det",type=int,default=300)
    a=ap.parse_args()
    repo=Path(a.repo); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for pp in ["nms","no-nms"]:
        name=f"week24_{a.tag}_{pp.replace('-','_')}_c001"
        cmd=[sys.executable,"val_e2e.py","--data",a.data,"--img",str(a.imgsz),
             "--batch",str(a.batch),"--weights",a.weights,"--device","0",
             "--name",name,"--postprocess",pp,"--conf-thres",str(a.conf),
             "--iou-thres",str(a.iou),"--max-det",str(a.max_det)]
        print("$"," ".join(cmd))
        r=subprocess.run(cmd,cwd=repo,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        print(r.stdout)
        (out/f"{name}.log").write_text(r.stdout or "")
        if r.returncode!=0: raise RuntimeError(f"{pp} failed")
        mp=latest(repo,name); d=json.loads(mp.read_text())
        rows.append({"tag":a.tag,"postprocess":pp,"conf_thres":a.conf,**d})
    pd.DataFrame(rows).to_csv(out/f"{a.tag}_e2e.csv",index=False)

if __name__=="__main__": main()
