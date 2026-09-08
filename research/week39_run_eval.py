
from __future__ import annotations
import argparse,json,re,subprocess,sys
from pathlib import Path

INTERNAL=re.compile(r"\ball\s+\d+\s+\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)")
AP95=re.compile(r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=\s*100\s*\]\s*=\s*([0-9.]+)")
AP50=re.compile(r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50\s*\|\s*area=\s*all\s*\|\s*maxDets=\s*100\s*\]\s*=\s*([0-9.]+)")
AP75=re.compile(r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.75\s*\|\s*area=\s*all\s*\|\s*maxDets=\s*100\s*\]\s*=\s*([0-9.]+)")

def one(rx,t,label):
    x=rx.findall(t)
    if len(x)!=1:raise RuntimeError(f"{label}: {x}")
    return float(x[0])

def main():
    a=argparse.ArgumentParser()
    for k in ["repo","script","weights","data","project","name","out","log"]:
        a.add_argument("--"+k,required=True)
    n=a.parse_args()
    cmd=[sys.executable,n.script,"--data",n.data,"--weights",n.weights,
         "--batch-size","16","--img","640","--conf-thres","0.001",
         "--iou-thres","0.7","--max-det","300","--device","0","--workers","2",
         "--save-json","--project",n.project,"--name",n.name,"--exist-ok"]
    p=subprocess.run(cmd,cwd=n.repo,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    Path(n.log).write_text(p.stdout)
    print(p.stdout)
    if p.returncode!=0 or "Traceback" in p.stdout:raise RuntimeError(n.log)
    internal=None
    for line in p.stdout.splitlines():
        m=INTERNAL.search(line)
        if m:
            internal={"precision":float(m.group(1)),"recall":float(m.group(2)),
                      "map50":float(m.group(3)),"map50_95":float(m.group(4))}
    if internal is None:raise RuntimeError("internal row missing")
    r={"internal":internal,"coco_api_map50_95":one(AP95,p.stdout,"AP95"),
       "coco_api_map50":one(AP50,p.stdout,"AP50"),"coco_api_ap75":one(AP75,p.stdout,"AP75")}
    Path(n.out).write_text(json.dumps(r,indent=2))
    print(json.dumps(r,indent=2))
if __name__=="__main__":main()
