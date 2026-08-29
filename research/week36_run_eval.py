from __future__ import annotations
import argparse,json,re,subprocess,sys
from pathlib import Path
INTERNAL_RE=re.compile(r"\ball\s+\d+\s+\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)")
AP95=re.compile(r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=\s*100\s*\]\s*=\s*([0-9.]+)")
AP50=re.compile(r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50\s*\|\s*area=\s*all\s*\|\s*maxDets=\s*100\s*\]\s*=\s*([0-9.]+)")
AP75=re.compile(r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.75\s*\|\s*area=\s*all\s*\|\s*maxDets=\s*100\s*\]\s*=\s*([0-9.]+)")
def one(rx,text,label):
    x=rx.findall(text)
    if len(x)!=1: raise RuntimeError(f"{label}: expected 1 hit, got {x}")
    return float(x[0])
def main():
    a=argparse.ArgumentParser()
    for k in ["repo","script","weights","data","project","name","out","log"]:
        a.add_argument("--"+k,required=True)
    a.add_argument("--conf",type=float,default=.001)
    ns=a.parse_args()
    cmd=[sys.executable,ns.script,"--data",ns.data,"--weights",ns.weights,
         "--batch-size","16","--img","640","--conf-thres",str(ns.conf),
         "--iou-thres","0.7","--max-det","300","--device","0","--workers","2",
         "--save-json","--project",ns.project,"--name",ns.name,"--exist-ok"]
    p=subprocess.run(cmd,cwd=ns.repo,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    Path(ns.log).parent.mkdir(parents=True,exist_ok=True)
    Path(ns.log).write_text(p.stdout,encoding="utf-8")
    print(p.stdout)
    if p.returncode!=0 or "Traceback" in p.stdout:
        raise RuntimeError(f"eval failed rc={p.returncode}: {ns.log}")
    internal=None
    for line in p.stdout.splitlines():
        m=INTERNAL_RE.search(line)
        if m:
            internal={"precision":float(m.group(1)),"recall":float(m.group(2)),
                      "map50":float(m.group(3)),"map50_95":float(m.group(4))}
    if internal is None: raise RuntimeError("internal metric row not found")
    r={"internal":internal,
       "coco_api_map50_95":one(AP95,p.stdout,"AP95"),
       "coco_api_map50":one(AP50,p.stdout,"AP50"),
       "coco_api_ap75":one(AP75,p.stdout,"AP75"),
       "conf":ns.conf,"iou_nms":.7,"max_det":300,"weights":ns.weights,"script":ns.script}
    Path(ns.out).write_text(json.dumps(r,indent=2),encoding="utf-8")
    print(json.dumps(r,indent=2))
if __name__=="__main__":
    main()
