
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from torchvision.ops import nms as tv_nms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.experimental import attempt_load
from utils.dataloaders import create_dataloader
from utils.general import check_img_size, non_max_suppression, scale_boxes, xywh2xyxy, xyxy2xywh, coco80_to_coco91_class
from utils.metrics import ap_per_class, box_iou
from utils.week46_nonms_postprocess import _to_b84n, yolov10_style_no_nms

def select_decoded(raw):
    if torch.is_tensor(raw):
        y = raw
    elif isinstance(raw, (list, tuple)) and len(raw) >= 1 and torch.is_tensor(raw[0]):
        y = raw[0]
    else:
        raise RuntimeError(f"Unsupported proxy output: {type(raw)}")
    return _to_b84n(y, nc=80)

def process_batch(detections, labels, iouv):
    correct = np.zeros((detections.shape[0], iouv.shape[0]), dtype=bool)
    if detections.shape[0] == 0 or labels.shape[0] == 0:
        return torch.tensor(correct, dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

def finalize_internal(stats):
    z = [torch.cat(x, 0).numpy() for x in zip(*stats)]
    if not len(z) or len(z[3]) == 0:
        return {"precision":0.0,"recall":0.0,"map50":0.0,"map50_95":0.0,"targets":0}
    if not z[0].any():
        return {"precision":0.0,"recall":0.0,"map50":0.0,"map50_95":0.0,"targets":int(len(z[3]))}
    r = ap_per_class(*z, plot=False, names={})
    p, rec, ap = r[2], r[3], r[5]
    return {
        "precision":float(np.mean(p)),
        "recall":float(np.mean(rec)),
        "map50":float(np.mean(ap[:,0])),
        "map50_95":float(np.mean(ap.mean(1))),
        "targets":int(len(z[3])),
    }

def diagnostic_sameclass_suppression(pred, iou_thres):
    if pred is None or len(pred) == 0:
        return pred
    max_wh = 7680.0
    boxes = pred[:, :4] + pred[:, 5:6] * max_wh
    keep = tv_nms(boxes, pred[:, 4], float(iou_thres))
    return pred[keep]

def duplicate_counts(pred, iou_thres=0.7):
    n = int(pred.shape[0])
    if n < 2:
        return 0, 0
    ious = box_iou(pred[:, :4], pred[:, :4])
    tri = torch.triu(torch.ones((n,n), device=pred.device, dtype=torch.bool), diagonal=1)
    overlap = (ious >= float(iou_thres)) & tri
    any_cls = int(overlap.sum().item())
    same = pred[:,5:6] == pred[:,5].view(1,-1)
    same_cls = int((overlap & same).sum().item())
    return same_cls, any_cls

def write_jsonl(fh, predn, path, class_map):
    if predn is None or len(predn) == 0:
        return
    image_id = int(Path(path).stem)
    box = xyxy2xywh(predn[:, :4].clone())
    box[:, :2] -= box[:, 2:] / 2
    for p, b in zip(predn.tolist(), box.tolist()):
        rec = {
            "image_id": image_id,
            "category_id": int(class_map[int(p[5])]),
            "bbox": [round(float(x),3) for x in b],
            "score": round(float(p[4]),6),
        }
        fh.write(json.dumps(rec, separators=(",",":")) + "\n")

def jsonl_to_json(src, dst):
    first = True
    with open(src,"r",encoding="utf-8") as fi, open(dst,"w",encoding="utf-8") as fo:
        fo.write("[")
        for line in fi:
            line = line.strip()
            if not line:
                continue
            if not first:
                fo.write(",")
            fo.write(line)
            first = False
        fo.write("]")

def coco_eval(annotation, pred_json, image_ids):
    gt = COCO(str(annotation))
    dt = gt.loadRes(str(pred_json))
    ev = COCOeval(gt, dt, "bbox")
    ev.params.imgIds = list(image_ids)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {
        "ap50_95":float(ev.stats[0]),
        "ap50":float(ev.stats[1]),
        "ap75":float(ev.stats[2]),
        "ar1":float(ev.stats[6]),
        "ar10":float(ev.stats[7]),
        "ar100":float(ev.stats[8]),
        "coco_standard_maxdets":list(map(int, ev.params.maxDets)),
    }

def init_method():
    return {"stats":[],"pred_count":0,"dup_same70":0,"dup_any70":0,"images":0,"targets":0}

def add_gt_diag(diag, labelsn, predn):
    if len(labelsn) == 0:
        return
    if predn is None or len(predn) == 0:
        for _ in range(len(labelsn)):
            diag["multiplicity_iou50"].append(0)
            diag["first_valid_rank_iou50"].append(None)
            diag["first_valid_score_iou50"].append(None)
        return
    order = torch.argsort(predn[:,4], descending=True)
    p = predn[order]
    ious = box_iou(labelsn[:,1:5], p[:,:4])
    pcls = p[:,5].long()
    for gi in range(len(labelsn)):
        same = pcls == labelsn[gi,0].long()
        if not bool(same.any()):
            diag["multiplicity_iou50"].append(0)
            diag["first_valid_rank_iou50"].append(None)
            diag["first_valid_score_iou50"].append(None)
            continue
        same_idx = torch.where(same)[0]
        same_ious = ious[gi, same_idx]
        valid = same_ious >= 0.50
        m = int(valid.sum().item())
        diag["multiplicity_iou50"].append(m)
        if m == 0:
            diag["first_valid_rank_iou50"].append(None)
            diag["first_valid_score_iou50"].append(None)
        else:
            valid_idx = same_idx[valid]
            first_idx = int(valid_idx.min().item())
            diag["first_valid_rank_iou50"].append(first_idx + 1)
            diag["first_valid_score_iou50"].append(float(p[first_idx,4].item()))

def summarize_gt_diag(diag):
    mult = np.asarray(diag["multiplicity_iou50"], dtype=np.int64)
    ranks = np.asarray([x for x in diag["first_valid_rank_iou50"] if x is not None], dtype=float)
    scores = np.asarray([x for x in diag["first_valid_score_iou50"] if x is not None], dtype=float)
    n = int(len(mult))
    hist = {
        "0":int((mult==0).sum()),
        "1":int((mult==1).sum()),
        "2":int((mult==2).sum()),
        "3_plus":int((mult>=3).sum()),
    }
    return {
        "gt_count":n,
        "multiplicity_iou50_histogram":hist,
        "multiplicity_iou50_fraction":{k:float(v/n) for k,v in hist.items()},
        "multiplicity_iou50_mean":float(mult.mean()),
        "multiplicity_iou50_median":float(np.median(mult)),
        "multiplicity_iou50_p95":float(np.percentile(mult,95)),
        "gt_with_valid_sameclass_iou50_rate":float((mult>0).mean()),
        "first_valid_rank_iou50_mean":float(ranks.mean()) if len(ranks) else None,
        "first_valid_rank_iou50_median":float(np.median(ranks)) if len(ranks) else None,
        "first_valid_rank_iou50_p95":float(np.percentile(ranks,95)) if len(ranks) else None,
        "first_valid_rank_le_10_rate_of_hit_gt":float((ranks<=10).mean()) if len(ranks) else None,
        "first_valid_rank_le_100_rate_of_hit_gt":float((ranks<=100).mean()) if len(ranks) else None,
        "first_valid_rank_le_300_rate_of_hit_gt":float((ranks<=300).mean()) if len(ranks) else None,
        "first_valid_score_iou50_mean":float(scores.mean()) if len(scores) else None,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--nms-iou", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    model = attempt_load(a.weights, device=device).float().eval()
    stride = int(model.stride.max().item())
    imgsz = int(check_img_size(a.imgsz, s=stride))

    loader = create_dataloader(
        a.val, imgsz, a.batch, stride,
        single_cls=False, pad=0.5, rect=True, workers=a.workers,
        prefix="week47-diagnostic: "
    )[0]
    assert len(loader.dataset.im_files) == 5000
    assert sum(len(x) for x in loader.dataset.labels) == 36335

    methods = [
        "A_native_nms","B_nonms_k300","C50_suppress","C70_suppress",
        "C90_suppress","K100_nonms","K500_nonms"
    ]
    state = {m:init_method() for m in methods}
    jsonl_paths = {m:out/f"{m}.jsonl" for m in methods}
    fhs = {m:open(jsonl_paths[m],"w",encoding="utf-8") for m in methods}
    gt_diag = {
        "multiplicity_iou50":[],
        "first_valid_rank_iou50":[],
        "first_valid_score_iou50":[],
    }

    iouv = torch.linspace(0.5,0.95,10,device=device)
    niou = len(iouv)
    class_map = coco80_to_coco91_class()
    forward_batches = 0

    try:
        with torch.inference_mode():
            for bi, (im, targets, paths, shapes) in enumerate(tqdm(loader, desc="Week47 O2O diagnostic")):
                im = im.to(device, non_blocking=True).float() / 255.0
                targets = targets.to(device)
                nb, _, h, w = im.shape
                decoded = select_decoded(model(im))
                forward_batches += 1

                A = non_max_suppression(
                    decoded.clone(), a.conf, a.nms_iou,
                    labels=(), multi_label=True, agnostic=False, max_det=300
                )
                B = yolov10_style_no_nms(decoded.clone(), a.conf, 300, 80)
                K100 = yolov10_style_no_nms(decoded.clone(), a.conf, 100, 80)
                K500 = yolov10_style_no_nms(decoded.clone(), a.conf, 500, 80)
                C50 = [diagnostic_sameclass_suppression(x,0.50) for x in B]
                C70 = [diagnostic_sameclass_suppression(x,0.70) for x in B]
                C90 = [diagnostic_sameclass_suppression(x,0.90) for x in B]
                outputs = {
                    "A_native_nms":A, "B_nonms_k300":B,
                    "C50_suppress":C50, "C70_suppress":C70, "C90_suppress":C90,
                    "K100_nonms":K100, "K500_nonms":K500,
                }

                targets_px = targets.clone()
                targets_px[:,2:] *= torch.tensor((w,h,w,h), device=device, dtype=targets_px.dtype)

                for si in range(nb):
                    labels = targets_px[targets_px[:,0] == si, 1:]
                    nl = int(labels.shape[0])
                    original_shape = shapes[si][0]

                    if nl:
                        tbox = xywh2xyxy(labels[:,1:5])
                        scale_boxes(im[si].shape[1:], tbox, original_shape, shapes[si][1])
                        labelsn = torch.cat((labels[:,0:1], tbox),1)
                    else:
                        labelsn = torch.empty((0,5), device=device, dtype=im.dtype)

                    Bn = B[si].clone()
                    if len(Bn):
                        scale_boxes(im[si].shape[1:], Bn[:,:4], original_shape, shapes[si][1])
                    add_gt_diag(gt_diag, labelsn, Bn)

                    for method in methods:
                        pred = outputs[method][si]
                        npr = int(pred.shape[0])
                        predn = pred.clone()
                        if npr:
                            scale_boxes(im[si].shape[1:], predn[:,:4], original_shape, shapes[si][1])

                        correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)
                        if nl and npr:
                            correct = process_batch(predn, labelsn, iouv)

                        state[method]["stats"].append((
                            correct.detach().cpu(),
                            pred[:,4].detach().float().cpu(),
                            pred[:,5].detach().float().cpu(),
                            labels[:,0].detach().float().cpu(),
                        ))

                        if npr:
                            ds, da = duplicate_counts(pred,0.70)
                            state[method]["dup_same70"] += ds
                            state[method]["dup_any70"] += da

                        state[method]["pred_count"] += npr
                        state[method]["images"] += 1
                        state[method]["targets"] += nl
                        write_jsonl(fhs[method], predn.detach().float().cpu(), paths[si], class_map)

                del im, targets, decoded, A, B, K100, K500, C50, C70, C90, outputs
                if (bi+1) % 50 == 0:
                    torch.cuda.empty_cache()
    finally:
        for fh in fhs.values():
            fh.close()

    assert forward_batches == len(loader)
    image_ids = [int(Path(x).stem) for x in loader.dataset.im_files]
    assert len(image_ids) == 5000 and len(set(image_ids)) == 5000

    results = {}
    for method in methods:
        s = state[method]
        assert s["images"] == 5000
        assert s["targets"] == 36335
        internal = finalize_internal(s["stats"])
        pred_json = out/f"{method}.json"
        jsonl_to_json(jsonl_paths[method], pred_json)
        coco = coco_eval(a.annotation, pred_json, image_ids)
        results[method] = {
            "internal":internal,
            "coco_api":coco,
            "prediction_count":int(s["pred_count"]),
            "predictions_per_image":float(s["pred_count"]/s["images"]),
            "sameclass_duplicate_pairs_iou70":int(s["dup_same70"]),
            "sameclass_duplicate_pairs_iou70_per_image":float(s["dup_same70"]/s["images"]),
            "anyclass_duplicate_pairs_iou70":int(s["dup_any70"]),
            "anyclass_duplicate_pairs_iou70_per_image":float(s["dup_any70"]/s["images"]),
            "prediction_json":str(pred_json),
        }

    payload = {
        "meta":{
            "weights":a.weights,
            "batch":a.batch,
            "imgsz":imgsz,
            "conf":a.conf,
            "native_nms_iou":a.nms_iou,
            "precision":"FP32",
            "one_forward_per_batch":True,
            "same_decoded_tensor_for_all_paths":True,
            "training_performed":False,
        },
        "results":results,
        "gt_diagnostics_B_nonms_k300":summarize_gt_diag(gt_diag),
    }
    (out/"week47_full_diagnostic.json").write_text(
        json.dumps(payload,indent=2), encoding="utf-8"
    )
    print(json.dumps(payload,indent=2))

if __name__ == "__main__":
    main()
