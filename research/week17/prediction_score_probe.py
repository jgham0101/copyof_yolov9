
import argparse
import csv
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
    from utils.e2e_postprocess import select_one2one, v10_no_nms_postprocess, class_aware_nms_postprocess
except Exception as e:
    select_one2one = None
    v10_no_nms_postprocess = None
    class_aware_nms_postprocess = None
    IMPORT_ERROR = repr(e)
else:
    IMPORT_ERROR = None

def image_paths_from_data(data_yaml, sample_n):
    import yaml
    data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    root = Path(data.get("path", ""))
    train = data.get("train", "")
    img_root = Path(train)
    if not img_root.is_absolute():
        img_root = root / train
    imgs = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        imgs.extend(sorted(img_root.glob(ext)))
    return imgs[:sample_n]

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

def tensor_stats(x):
    if not torch.is_tensor(x):
        return {"raw_type": str(type(x))}
    t = x.detach().float().cpu()
    out = {
        "raw_shape": list(t.shape),
        "raw_min": float(t.min()) if t.numel() else None,
        "raw_max": float(t.max()) if t.numel() else None,
        "raw_mean": float(t.mean()) if t.numel() else None,
    }
    if t.ndim == 3:
        # Common detection tensor forms: [B, N, C] or [B, C, N].
        a = t[0]
        if a.shape[-1] >= 6:
            conf_candidate = a[:, 4]
            out.update({
                "candidate_axis": "last_dim",
                "candidate_count": int(a.shape[0]),
                "conf_like_mean": float(conf_candidate.mean()),
                "conf_like_max": float(conf_candidate.max()),
                "conf_like_p95": float(torch.quantile(conf_candidate, 0.95)),
                "conf_like_gt_0p001": int((conf_candidate > 0.001).sum()),
                "conf_like_gt_0p01": int((conf_candidate > 0.01).sum()),
                "conf_like_gt_0p1": int((conf_candidate > 0.1).sum()),
            })
        elif a.shape[0] >= 6:
            conf_candidate = a[4, :]
            out.update({
                "candidate_axis": "channel_dim",
                "candidate_count": int(a.shape[1]),
                "conf_like_mean": float(conf_candidate.mean()),
                "conf_like_max": float(conf_candidate.max()),
                "conf_like_p95": float(torch.quantile(conf_candidate, 0.95)),
                "conf_like_gt_0p001": int((conf_candidate > 0.001).sum()),
                "conf_like_gt_0p01": int((conf_candidate > 0.01).sum()),
                "conf_like_gt_0p1": int((conf_candidate > 0.1).sum()),
            })
    return out

def as_det(x):
    if x is None:
        return torch.empty((0, 6))
    if torch.is_tensor(x):
        return x[0] if x.ndim == 3 else x
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return torch.empty((0, 6))
        if torch.is_tensor(x[0]):
            t = x[0]
            return t[0] if t.ndim == 3 else t
        for v in x:
            t = as_det(v)
            if t is not None:
                return t
    return torch.empty((0, 6))

def det_stats(det):
    if det is None or len(det) == 0:
        return {"num_pred": 0, "conf_min": None, "conf_mean": None, "conf_max": None}
    det = det.detach().cpu()
    conf = det[:, 4].float()
    return {
        "num_pred": int(len(det)),
        "conf_min": float(conf.min()),
        "conf_mean": float(conf.mean()),
        "conf_max": float(conf.max()),
        "conf_p95": float(torch.quantile(conf, 0.95)),
    }

def postprocess(raw, mode, conf, iou, max_det):
    if mode == "native_nms":
        pred = raw[0] if isinstance(raw, (list, tuple)) else raw
        return non_max_suppression(pred, conf_thres=conf, iou_thres=iou, max_det=max_det)[0]
    if IMPORT_ERROR:
        raise RuntimeError(IMPORT_ERROR)
    one = select_one2one(raw)
    if mode == "one2one_nms":
        return as_det(class_aware_nms_postprocess(one, conf_thres=conf, iou_thres=iou, max_det=max_det))
    if mode == "one2one_no_nms":
        return as_det(v10_no_nms_postprocess(one, conf_thres=conf, iou_thres=iou, max_det=max_det))
    raise ValueError(mode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf-list", default="0.0001,0.001,0.01,0.1")
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--sample-n", type=int, default=32)
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

    imgs = image_paths_from_data(args.data, args.sample_n)
    assert imgs, "no sample images found"

    confs = [float(x) for x in args.conf_list.split(",") if x.strip()]
    rows = []
    raw_rows = []

    with torch.no_grad():
        for img_path in imgs:
            im, im0 = load_img(img_path, args.imgsz, stride)
            im = im.to(device)
            raw = model(im)

            raw_summary = {"tag": args.tag, "image": img_path.name}
            if select_one2one is not None:
                try:
                    one = select_one2one(raw)
                    raw_summary.update(tensor_stats(one))
                except Exception as e:
                    raw_summary["raw_error"] = repr(e)
            else:
                raw_summary["raw_error"] = IMPORT_ERROR
            raw_rows.append(raw_summary)

            for conf in confs:
                for mode in ["native_nms", "one2one_nms", "one2one_no_nms"]:
                    row = {"tag": args.tag, "image": img_path.name, "mode": mode, "conf_thres": conf}
                    try:
                        det = postprocess(raw, mode, conf, args.iou, args.max_det)
                        if det is not None and len(det):
                            det = det.detach().clone()
                            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                        row.update(det_stats(det))
                        row["error"] = ""
                    except Exception as e:
                        row.update({"num_pred": None, "conf_min": None, "conf_mean": None, "conf_max": None, "error": repr(e)})
                    rows.append(row)

    detail = pd.DataFrame(rows)
    raw_df = pd.DataFrame(raw_rows)

    detail.to_csv(out_dir / f"{args.tag}_prediction_probe_detail.csv", index=False)
    raw_df.to_csv(out_dir / f"{args.tag}_raw_one2one_summary.csv", index=False)

    agg = detail.groupby(["tag", "mode", "conf_thres"], dropna=False).agg(
        images=("image", "count"),
        mean_num_pred=("num_pred", "mean"),
        median_num_pred=("num_pred", "median"),
        max_num_pred=("num_pred", "max"),
        mean_conf=("conf_mean", "mean"),
        max_conf=("conf_max", "max"),
        errors=("error", lambda x: int((x.fillna("").astype(str) != "").sum())),
    ).reset_index()

    raw_agg = raw_df.drop(columns=["image"], errors="ignore").groupby(["tag"], dropna=False).mean(numeric_only=True).reset_index()

    agg.to_csv(out_dir / f"{args.tag}_prediction_probe_aggregate.csv", index=False)
    raw_agg.to_csv(out_dir / f"{args.tag}_raw_one2one_aggregate.csv", index=False)

    print(agg)
    print(raw_agg)
    print("saved:", out_dir)

if __name__ == "__main__":
    main()
