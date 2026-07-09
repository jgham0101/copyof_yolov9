
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes

try:
    from utils.e2e_postprocess import select_one2one, class_aware_nms_postprocess, v10_no_nms_postprocess
    IMPORT_ERROR = None
except Exception as e:
    select_one2one = None
    class_aware_nms_postprocess = None
    v10_no_nms_postprocess = None
    IMPORT_ERROR = repr(e)

def shape_desc(x, depth=0):
    if depth > 4:
        return "..."
    if torch.is_tensor(x):
        return {"type": "tensor", "shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, dict):
        return {"type": "dict", "keys": list(x.keys()), "values": {str(k): shape_desc(v, depth+1) for k, v in x.items()}}
    if isinstance(x, (list, tuple)):
        return {"type": type(x).__name__, "len": len(x), "items": [shape_desc(v, depth+1) for v in list(x)[:6]]}
    return {"type": str(type(x))}

def image_paths_from_data(data_yaml, sample_n):
    import yaml
    data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    root = Path(data.get("path", ""))
    train = data.get("val", data.get("train", ""))
    img_root = Path(train)
    if not img_root.is_absolute():
        img_root = root / train
    imgs = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        imgs.extend(sorted(img_root.glob(ext)))
    return imgs[:sample_n], root

def label_path_for_image(img_path):
    p = Path(img_path)
    parts = list(p.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        lab = Path(*parts).with_suffix(".txt")
        return lab
    return p.with_suffix(".txt")

def load_gt(img_path, im0_shape):
    h, w = im0_shape[:2]
    lab = label_path_for_image(img_path)
    rows = []
    if not lab.exists():
        return np.zeros((0, 5), dtype=np.float32)
    for line in lab.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        vals = [float(x) for x in line.split()[:5]]
        cls, xc, yc, bw, bh = vals
        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w
        y2 = (yc + bh / 2) * h
        rows.append([x1, y1, x2, y2, cls])
    return np.array(rows, dtype=np.float32)

def box_iou_np(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.clip(br - tl, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return inter / union

def load_img(path, imgsz, stride=32):
    im0 = cv2.imread(str(path))
    if im0 is None:
        raise FileNotFoundError(path)
    im = letterbox(im0, imgsz, stride=stride, auto=True)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).float() / 255.0
    if im.ndim == 3:
        im = im.unsqueeze(0)
    return im, im0

def combine_boxes_scores_labels(boxes, scores, labels):
    if not torch.is_tensor(boxes) or not torch.is_tensor(scores):
        return None
    b = boxes.detach()
    s = scores.detach()
    l = labels.detach() if torch.is_tensor(labels) else torch.zeros_like(s)
    if b.ndim == 3:
        b = b[0]
    if s.ndim > 1:
        s = s.reshape(-1)
    if l.ndim > 1:
        l = l.reshape(-1)
    if b.ndim == 2 and b.shape[-1] == 4 and s.numel() == b.shape[0]:
        return torch.cat([b.float(), s[:, None].float(), l[:, None].float()], dim=1)
    return None

def normalize_det(obj, batch_index=0):
    """Return Nx6 tensor [x1,y1,x2,y2,conf,cls] if any plausible detection tensor exists."""
    if obj is None:
        return torch.zeros((0, 6))

    if torch.is_tensor(obj):
        t = obj.detach()
        if t.ndim == 3:
            if t.shape[0] > batch_index:
                t = t[batch_index]
            elif t.shape[1] > batch_index:
                t = t[:, batch_index]
        if t.ndim == 2:
            # Nx6 or Nx(>=6)
            if t.shape[1] >= 6:
                return t[:, :6].float()
            # 6xN or CxN
            if t.shape[0] >= 6 and t.shape[1] > t.shape[0]:
                return t[:6, :].T.float()
        if t.ndim == 1 and t.numel() >= 6:
            return t[:6].reshape(1, 6).float()
        return torch.zeros((0, 6))

    if isinstance(obj, dict):
        # common keys
        for keys in [("boxes", "scores", "labels"), ("bboxes", "scores", "labels")]:
            if all(k in obj for k in keys):
                out = combine_boxes_scores_labels(obj[keys[0]], obj[keys[1]], obj[keys[2]])
                if out is not None:
                    return out
        for k in ["pred", "detections", "det", "output", "y"]:
            if k in obj:
                out = normalize_det(obj[k], batch_index)
                if len(out):
                    return out
        return torch.zeros((0, 6))

    if isinstance(obj, (list, tuple)):
        # possible tuple/list of boxes, scores, labels
        if len(obj) == 3:
            out = combine_boxes_scores_labels(obj[0], obj[1], obj[2])
            if out is not None:
                return out
        # common case: list length == batch, each element Nx6
        if len(obj) > batch_index:
            out = normalize_det(obj[batch_index], 0)
            if len(out):
                return out
        # recursive search
        for v in obj:
            out = normalize_det(v, batch_index)
            if len(out):
                return out
        return torch.zeros((0, 6))

    return torch.zeros((0, 6))


def call_v10_no_nms_safely(one, conf, iou, max_det):
    """
    Robust wrapper for v10_no_nms_postprocess.

    Different local implementations may use different signatures.
    We try the safest likely forms and then apply conf/max_det filtering
    after normalization.
    """
    errors = []

    call_patterns = [
        ((), {"conf_thres": conf, "max_det": max_det}),
        ((), {"conf": conf, "max_det": max_det}),
        ((), {"max_det": max_det, "conf_thres": conf}),
        ((), {"max_det": max_det}),
        ((conf, max_det), {}),
        ((conf,), {"max_det": max_det}),
        ((), {}),
    ]

    out = None
    success = False

    for args, kwargs in call_patterns:
        try:
            out = v10_no_nms_postprocess(one, *args, **kwargs)
            success = True
            break
        except Exception as e:
            errors.append(f"args={args}, kwargs={kwargs}, error={repr(e)}")

    if not success:
        raise RuntimeError("v10_no_nms_postprocess failed for all signatures: " + " | ".join(errors[:5]))

    det = normalize_det(out)

    # Apply confidence filtering and max_det here, independent of postprocess signature.
    if det is not None and len(det):
        det = det.float()
        det = det[det[:, 4] >= float(conf)]
        if len(det) > int(max_det):
            order = torch.argsort(det[:, 4], descending=True)[:int(max_det)]
            det = det[order]

    return det


def postprocess(raw, mode, conf, iou, max_det):
    if mode == "native_nms":
        pred = raw[0] if isinstance(raw, (list, tuple)) else raw
        return non_max_suppression(pred, conf_thres=conf, iou_thres=iou, max_det=max_det)[0]

    if IMPORT_ERROR:
        raise RuntimeError(IMPORT_ERROR)

    one = select_one2one(raw)

    if mode == "one2one_nms":
        out = class_aware_nms_postprocess(one, conf_thres=conf, iou_thres=iou, max_det=max_det)
        det = normalize_det(out)
        if det is not None and len(det):
            det = det.float()
            det = det[det[:, 4] >= float(conf)]
            if len(det) > int(max_det):
                order = torch.argsort(det[:, 4], descending=True)[:int(max_det)]
                det = det[order]
        return det

    if mode == "one2one_no_nms":
        return call_v10_no_nms_safely(one, conf, iou, max_det)

    raise ValueError(mode)

def summarize_det(det, gt):
    if det is None:
        det = torch.zeros((0, 6))
    if torch.is_tensor(det):
        det_np = det.detach().cpu().numpy()
    else:
        det_np = np.asarray(det)
    if det_np.ndim != 2 or det_np.shape[1] < 6:
        det_np = np.zeros((0, 6), dtype=np.float32)
    out = {
        "num_pred": int(len(det_np)),
        "num_gt": int(len(gt)),
        "mean_conf": None,
        "max_conf": None,
        "mean_best_iou": None,
        "match_iou50": 0,
        "class_match_iou50": 0,
        "duplicate_pairs_iou70_same_class": 0,
    }
    if len(det_np) == 0:
        return out

    conf = det_np[:, 4].astype(np.float32)
    out["mean_conf"] = float(conf.mean())
    out["max_conf"] = float(conf.max())

    if len(gt):
        ious = box_iou_np(det_np[:, :4], gt[:, :4])
        best_idx = ious.argmax(axis=1)
        best = ious.max(axis=1)
        pred_cls = det_np[:, 5].astype(np.int64)
        gt_cls = gt[best_idx, 4].astype(np.int64)
        out["mean_best_iou"] = float(best.mean())
        out["match_iou50"] = int((best >= 0.5).sum())
        out["class_match_iou50"] = int(((best >= 0.5) & (pred_cls == gt_cls)).sum())
    if len(det_np) > 1:
        iou_pred = box_iou_np(det_np[:, :4], det_np[:, :4])
        cls = det_np[:, 5].astype(np.int64)
        dup = 0
        for i in range(len(det_np)):
            for j in range(i + 1, len(det_np)):
                if cls[i] == cls[j] and iou_pred[i, j] >= 0.7:
                    dup += 1
        out["duplicate_pairs_iou70_same_class"] = int(dup)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf-list", default="0.0001,0.001,0.005,0.01,0.05")
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--sample-n", type=int, default=64)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = attempt_load(args.weights, device=device)
    model.eval()

    stride = 32
    if hasattr(model, "stride"):
        try:
            stride = int(model.stride.max())
        except Exception:
            pass

    imgs, _ = image_paths_from_data(args.data, args.sample_n)
    assert imgs, "no images found"

    confs = [float(x) for x in args.conf_list.split(",") if x.strip()]
    detail_rows = []
    shape_rows = []

    with torch.no_grad():
        for image_i, img_path in enumerate(imgs):
            im, im0 = load_img(img_path, args.imgsz, stride)
            im = im.to(device)
            gt = load_gt(img_path, im0.shape)
            raw = model(im)

            one_shape = None
            try:
                one_shape = shape_desc(select_one2one(raw)) if select_one2one is not None else {"error": IMPORT_ERROR}
            except Exception as e:
                one_shape = {"error": repr(e)}

            shape_rows.append({
                "tag": args.tag,
                "image": img_path.name,
                "raw_shape": json.dumps(shape_desc(raw), ensure_ascii=False),
                "one2one_shape": json.dumps(one_shape, ensure_ascii=False),
            })

            for conf in confs:
                det_by_mode = {}
                for mode in ["native_nms", "one2one_nms", "one2one_no_nms"]:
                    row = {
                        "tag": args.tag,
                        "image": img_path.name,
                        "image_index": image_i,
                        "mode": mode,
                        "conf_thres": conf,
                        "max_det": args.max_det,
                    }
                    try:
                        det = postprocess(raw, mode, conf, args.iou, args.max_det)
                        if det is not None and len(det):
                            det = det.detach().clone()
                            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                        det_by_mode[mode] = det
                        row.update(summarize_det(det, gt))
                        row["error"] = ""
                    except Exception as e:
                        det_by_mode[mode] = torch.zeros((0, 6))
                        row.update(summarize_det(torch.zeros((0, 6)), gt))
                        row["error"] = repr(e)
                    detail_rows.append(row)

    detail = pd.DataFrame(detail_rows)
    shape_df = pd.DataFrame(shape_rows)

    detail_path = out_dir / f"{args.tag}_nonms_probe_detail.csv"
    shape_path = out_dir / f"{args.tag}_output_shapes.csv"
    detail.to_csv(detail_path, index=False)
    shape_df.to_csv(shape_path, index=False)

    agg = detail.groupby(["tag", "mode", "conf_thres", "max_det"], dropna=False).agg(
        images=("image", "count"),
        mean_num_pred=("num_pred", "mean"),
        median_num_pred=("num_pred", "median"),
        max_num_pred=("num_pred", "max"),
        mean_conf=("mean_conf", "mean"),
        max_conf=("max_conf", "max"),
        mean_best_iou=("mean_best_iou", "mean"),
        total_match_iou50=("match_iou50", "sum"),
        total_class_match_iou50=("class_match_iou50", "sum"),
        total_duplicate_pairs_iou70_same_class=("duplicate_pairs_iou70_same_class", "sum"),
        errors=("error", lambda x: int((x.fillna("").astype(str) != "").sum())),
    ).reset_index()

    agg_path = out_dir / f"{args.tag}_nonms_probe_aggregate.csv"
    agg.to_csv(agg_path, index=False)

    print("aggregate")
    print(agg)
    print("saved:", detail_path, agg_path, shape_path)

if __name__ == "__main__":
    main()
