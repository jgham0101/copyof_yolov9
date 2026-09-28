
import argparse, importlib.util, json, os
from collections import Counter
from pathlib import Path
import numpy as np
import torch, yaml

from models.experimental import attempt_load
from utils.dataloaders import create_dataloader
from utils.general import check_img_size, non_max_suppression
from utils.metrics import box_iou
from utils.loss_tal_week48_duplicate_hn import ComputeLossWeek28V2 as CaptureLoss

ROOT=Path(__file__).resolve().parents[1]
W49_SCRIPT=ROOT/"research/week49_selector_coverage_one_negative_diagnostic.py"
spec=importlib.util.spec_from_file_location("w49",W49_SCRIPT)
w49=importlib.util.module_from_spec(spec)
spec.loader.exec_module(w49)

def positive_class_maps(loss_fn):
    cap=loss_fn._week48_assigner_last
    out=cap["output"]
    target_scores=out[2].detach()
    fg_mask=out[3].bool().detach()
    result=[]
    for bi in range(target_scores.shape[0]):
        fg=torch.where(fg_mask[bi])[0]
        tc={int(a.item()):int(target_scores[bi,int(a.item())].argmax().item()) for a in fg}
        result.append({"fg_anchors":set(tc.keys()),"target_cls":tc})
    return result

def connected_components_sameclass(pred,anchors,assignment,iou_thr=.70):
    comps=[]
    if pred.shape[0]<=1:return comps
    classes=pred[:,5].long()
    for cls_t in classes.unique():
        cls=int(cls_t.item())
        ids=torch.where(classes==cls_t)[0]
        if ids.numel()<=1:continue
        ious=box_iou(pred[ids,:4],pred[ids,:4])
        adj=(ious>=iou_thr).detach().cpu().numpy()
        ids_cpu=ids.detach().cpu().tolist()
        visited=set()
        for st in range(len(ids_cpu)):
            if st in visited:continue
            stack=[st];visited.add(st);local=[]
            while stack:
                u=stack.pop();local.append(u)
                for v in np.where(adj[u])[0].tolist():
                    if v not in visited:
                        visited.add(v);stack.append(v)
            if len(local)<=1:continue
            glob=[ids_cpu[x] for x in local]
            order=torch.argsort(pred[glob,4],descending=True).detach().cpu().tolist()
            ranked=[glob[int(x)] for x in order]
            anchor_ids=[int(anchors[g].item()) for g in glob]
            fg_count=sum(a in assignment["fg_anchors"] for a in anchor_ids)
            samepos=sum(
                a in assignment["target_cls"] and assignment["target_cls"][a]==cls
                for a in anchor_ids
            )
            rep_anchor=int(anchors[ranked[0]].item())
            if rep_anchor not in assignment["fg_anchors"]:
                rep_status="UNMATCHED"
            elif assignment["target_cls"][rep_anchor]==cls:
                rep_status="MATCHED_SAME_CLASS"
            else:
                rep_status="MATCHED_OTHER_CLASS"
            sub=adj[np.ix_(local,local)]
            edges=int(np.triu(sub,k=1).sum())
            sc=pred[ranked,4].detach().float().cpu().numpy()
            comps.append({
                "cls":cls,"ranked_nodes":ranked,"size":len(glob),"edge_count":edges,
                "fg_count":int(fg_count),"sameclass_positive_count":int(samepos),
                "strict_unmatched_only":fg_count==0,
                "zero_sameclass_positive":samepos==0,
                "representative_status":rep_status,
                "top1_score":float(sc[0]),
                "top1_top2_margin":float(sc[0]-sc[1]),
            })
    return comps

def masks_from_components(pred,comps):
    n=int(pred.shape[0])
    names=["R1_remove_1","R2_remove_2","R3_remove_3","KEEP1","ZERO_SAMEPOS_KEEP1"]
    masks={k:torch.ones(n,dtype=torch.bool,device=pred.device) for k in names}
    for c in comps:
        rivals=c["ranked_nodes"][1:]
        for rid in rivals[:1]:masks["R1_remove_1"][rid]=False
        for rid in rivals[:2]:masks["R2_remove_2"][rid]=False
        for rid in rivals[:3]:masks["R3_remove_3"][rid]=False
        for rid in rivals:masks["KEEP1"][rid]=False
        if c["zero_sameclass_positive"]:
            for rid in rivals:masks["ZERO_SAMEPOS_KEEP1"][rid]=False
    return masks

def update(acc,comps):
    for c in comps:
        acc["component_count"]+=1
        acc["prediction_count"]+=c["size"]
        acc["edge_count"]+=c["edge_count"]
        sb="2" if c["size"]==2 else "3" if c["size"]==3 else "4" if c["size"]==4 else "5_plus"
        fb="0" if c["fg_count"]==0 else "1" if c["fg_count"]==1 else "2_plus"
        sp=c["sameclass_positive_count"]
        pb="0" if sp==0 else "1" if sp==1 else "2_plus"
        acc["size_hist"][sb]+=1;acc["fg_hist"][fb]+=1;acc["samepos_hist"][pb]+=1
        acc["rep_status"][c["representative_status"]]+=1
        acc["strict_unmatched"]+=int(c["strict_unmatched_only"])
        acc["zero_samepos"]+=int(c["zero_sameclass_positive"])
        acc["sizes"].append(c["size"])
        acc["margins"].append(c["top1_top2_margin"])

def finalize(acc,images):
    n=acc["component_count"]
    def frac(counter):return {k:(v/n if n else 0.) for k,v in counter.items()}
    return {
        "images":images,"component_count":n,"components_per_image":n/images,
        "predictions_in_components":acc["prediction_count"],
        "duplicate_graph_edges":acc["edge_count"],
        "size_hist":dict(acc["size_hist"]),"size_fraction":frac(acc["size_hist"]),
        "fg_hist":dict(acc["fg_hist"]),"fg_fraction":frac(acc["fg_hist"]),
        "sameclass_positive_hist":dict(acc["samepos_hist"]),
        "sameclass_positive_fraction":frac(acc["samepos_hist"]),
        "strict_unmatched_only_count":acc["strict_unmatched"],
        "strict_unmatched_only_fraction":acc["strict_unmatched"]/n if n else 0.,
        "zero_sameclass_positive_count":acc["zero_samepos"],
        "zero_sameclass_positive_fraction":acc["zero_samepos"]/n if n else 0.,
        "representative_status":dict(acc["rep_status"]),
        "representative_status_fraction":frac(acc["rep_status"]),
        "size_mean":float(np.mean(acc["sizes"])) if acc["sizes"] else None,
        "size_median":float(np.median(acc["sizes"])) if acc["sizes"] else None,
        "size_p95":float(np.percentile(acc["sizes"],95)) if acc["sizes"] else None,
        "top1_top2_margin_mean":float(np.mean(acc["margins"])) if acc["margins"] else None,
        "top1_top2_margin_median":float(np.median(acc["margins"])) if acc["margins"] else None,
    }

def evaluate(label,raw_weight,proxy_weight,val_txt,annotation,out_dir,hyp_path,
             batch=8,imgsz=640,conf=.001,max_det=300,iou=.70,workers=2,
             max_images=0,skip_coco=False):
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda:0")
    proxy=attempt_load(str(proxy_weight),device=device,inplace=True,fuse=False).float().eval()
    raw=attempt_load(str(raw_weight),device=device,inplace=True,fuse=False).float()
    raw.hyp=dict(yaml.safe_load(Path(hyp_path).read_text()))
    raw.train()
    for m in raw.modules():
        if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
    if hasattr(raw.model[-1],"week28_dual_training"):raw.model[-1].week28_dual_training=True
    os.environ["YOLO_WEEK48_DUP_HN_IOU"]=str(iou)
    os.environ["YOLO_WEEK48_DUP_HN_LAMBDA"]="0.0"
    os.environ["YOLO_WEEK48_DUP_HN_TELEMETRY"]=""
    capture=CaptureLoss(raw)

    stride=int(proxy.stride.max().item())
    imgsz=int(check_img_size(imgsz,s=stride))
    loader=create_dataloader(str(val_txt),imgsz,batch,stride,False,pad=.5,rect=True,
                             workers=workers,prefix=f"week50-{label}: ")[0]

    methods=["A_native_nms","B_nonms_k300","R1_remove_1","R2_remove_2",
             "R3_remove_3","KEEP1","ZERO_SAMEPOS_KEEP1","C70_diagnostic"]
    writers={};paths={}
    if not skip_coco:
        for m in methods:
            paths[m]=out_dir/f"{m}.json"
            writers[m]=w49.JsonArrayWriter(paths[m])
    cmap=w49.coco80_to_coco91_class()

    acc={
        "component_count":0,"prediction_count":0,"edge_count":0,
        "strict_unmatched":0,"zero_samepos":0,
        "size_hist":Counter(),"fg_hist":Counter(),"samepos_hist":Counter(),
        "rep_status":Counter(),"sizes":[],"margins":[]
    }
    totals=Counter();align_max=0.;align_means=[];checked=False

    try:
        for im,targets,impaths,shapes in loader:
            if max_images and totals["images"]>=max_images:break
            im=im.to(device).float()/255.;targets=targets.to(device)
            with torch.inference_mode():
                raw_out=raw(im)
                _=capture(raw_out,targets,img=im,epoch=0)
                decoded=w49.extract_proxy_decoded(proxy(im))
            logits=capture._week48_bce.logits.detach()
            ps=decoded[:,4:,:].permute(0,2,1).contiguous()
            diff=(logits.sigmoid()-ps).abs()
            align_max=max(align_max,float(diff.max()));align_means.append(float(diff.mean()))
            if not checked:
                if not torch.allclose(logits.sigmoid(),ps,atol=5e-4,rtol=5e-4):
                    raise RuntimeError(f"Anchor alignment failed max={float(diff.max())}")
                checked=True

            amap=positive_class_maps(capture)
            indexed=w49.indexed_no_nms(decoded,conf=conf,max_det=max_det,nc=80)
            A=non_max_suppression(decoded,conf_thres=conf,iou_thres=iou,classes=None,
                                  agnostic=False,multi_label=True,labels=(),max_det=max_det)

            for bi in range(len(impaths)):
                if max_images and totals["images"]>=max_images:break
                B=indexed[bi]["pred"];anchors=indexed[bi]["anchor"]
                comps=connected_components_sameclass(B,anchors,amap[bi],iou)
                update(acc,comps)
                masks=masks_from_components(B,comps)
                variants={
                    "A_native_nms":A[bi],"B_nonms_k300":B,
                    "R1_remove_1":B[masks["R1_remove_1"]],
                    "R2_remove_2":B[masks["R2_remove_2"]],
                    "R3_remove_3":B[masks["R3_remove_3"]],
                    "KEEP1":B[masks["KEEP1"]],
                    "ZERO_SAMEPOS_KEEP1":B[masks["ZERO_SAMEPOS_KEEP1"]],
                    "C70_diagnostic":w49.class_aware_c70(B,iou=iou),
                }
                totals["images"]+=1
                if not skip_coco:
                    for name,pred in variants.items():
                        w49.write_preds(writers[name],pred,impaths[bi],shapes[bi],im.shape[2:],cmap)
    finally:
        for w in writers.values():w.close()

    topology=finalize(acc,totals["images"])
    methodout={}
    if not skip_coco:
        for m in methods:
            methodout[m]={
                "prediction_count":writers[m].count,
                "predictions_per_image":writers[m].count/totals["images"],
                "coco_api":w49.coco_eval(annotation,paths[m]),
                "prediction_json":str(paths[m]),
            }
    result={
        "label":label,"images":totals["images"],
        "raw_weight":str(raw_weight),"proxy_weight":str(proxy_weight),
        "config":{"batch":batch,"imgsz":imgsz,"conf":conf,"max_det":max_det,
                  "iou":iou,"precision":"FP32","training_performed":False},
        "anchor_alignment":{
            "max_abs":align_max,"mean_abs":float(np.mean(align_means)),
            "tolerance":5e-4,"pass":checked
        },
        "component_topology":topology,
        "methods":methodout
    }
    (out_dir/"week50_diagnostic.json").write_text(json.dumps(result,indent=2))
    del raw,proxy,capture
    torch.cuda.empty_cache()
    return result

def main():
    ap=argparse.ArgumentParser()
    for x in ["label","raw-weight","proxy-weight","val","annotation","out-dir","hyp"]:
        ap.add_argument("--"+x,required=True)
    ap.add_argument("--batch",type=int,default=8)
    ap.add_argument("--imgsz",type=int,default=640)
    ap.add_argument("--conf",type=float,default=.001)
    ap.add_argument("--max-det",type=int,default=300)
    ap.add_argument("--iou",type=float,default=.70)
    ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--max-images",type=int,default=0)
    ap.add_argument("--skip-coco",action="store_true")
    a=ap.parse_args()
    r=evaluate(a.label,Path(a.raw_weight),Path(a.proxy_weight),Path(a.val),
               Path(a.annotation),Path(a.out_dir),Path(a.hyp),a.batch,a.imgsz,
               a.conf,a.max_det,a.iou,a.workers,a.max_images,a.skip_coco)
    print(json.dumps({
        "label":r["label"],"images":r["images"],
        "anchor_alignment":r["anchor_alignment"],
        "component_topology":r["component_topology"],
        "method_ap":{k:v["coco_api"]["ap50_95"] for k,v in r["methods"].items()}
                    if r["methods"] else {}
    },indent=2))

if __name__=="__main__":
    main()
