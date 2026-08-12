
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
from utils.general import scale_boxes
from utils.e2e_postprocess import select_one2one, v10_no_nms_postprocess


def image_paths_from_data(data_yaml, sample_n):
    import yaml
    data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    root = Path(data.get("path", ""))
    val = data.get("val", data.get("train", ""))
    img_root = Path(val)
    if not img_root.is_absolute():
        img_root = root / val

    imgs = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        imgs.extend(sorted(img_root.glob(ext)))

    return imgs[:sample_n]


def label_path_for_image(img_path):
    p = Path(img_path)
    parts = list(p.parts)

    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")

    return p.with_suffix(".txt")


def load_gt(img_path, shape):
    h, w = shape[:2]
    lab = label_path_for_image(img_path)
    rows = []

    if not lab.exists():
        return np.zeros((0, 5), dtype=np.float32)

    for line in lab.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        cls, xc, yc, bw, bh = [float(x) for x in line.split()[:5]]
        rows.append([
            (xc - bw / 2) * w,
            (yc - bh / 2) * h,
            (xc + bw / 2) * w,
            (yc + bh / 2) * h,
            cls,
        ])

    return np.asarray(rows, dtype=np.float32)


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


def box_iou_np(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.clip(br - tl, 0, None)

    inter = wh[..., 0] * wh[..., 1]
    aa = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    bb = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)

    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def normalize_det(obj):
    if obj is None:
        return torch.zeros((0, 6))

    if torch.is_tensor(obj):
        t = obj.detach()

        if t.ndim == 3:
            t = t[0]

        if t.ndim == 2 and t.shape[1] >= 6:
            return t[:, :6].float()

        if t.ndim == 2 and t.shape[0] >= 6 and t.shape[1] > t.shape[0]:
            return t[:6, :].T.float()

        return torch.zeros((0, 6))

    if isinstance(obj, (list, tuple)):
        if len(obj):
            return normalize_det(obj[0])

    if isinstance(obj, dict):
        for k in ["pred", "detections", "det", "output", "y"]:
            if k in obj:
                out = normalize_det(obj[k])
                if len(out):
                    return out

    return torch.zeros((0, 6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.0001)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--match-iou", type=float, default=0.3)
    ap.add_argument("--sample-n", type=int, default=128)
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
    assert imgs, "No validation images found."

    rows = []

    with torch.no_grad():
        for image_i, img_path in enumerate(imgs):
            im, im0 = load_img(img_path, args.imgsz, stride)
            im = im.to(device)
            gt = load_gt(img_path, im0.shape)

            raw = model(im)
            one = select_one2one(raw)
            det = normalize_det(
                v10_no_nms_postprocess(
                    one,
                    conf_thres=args.conf,
                    max_det=args.max_det,
                )
            )

            if len(det):
                det = det.clone()
                det[:, :4] = scale_boxes(
                    im.shape[2:],
                    det[:, :4],
                    im0.shape,
                )

            d = det.detach().cpu().numpy() if len(det) else np.zeros((0, 6), dtype=np.float32)

            if len(gt) == 0:
                continue

            iou = box_iou_np(gt[:, :4], d[:, :4]) if len(d) else np.zeros((len(gt), 0))

            for gi, g in enumerate(gt):
                gt_cls = int(g[4])

                if len(d):
                    cls_mask = d[:, 5].astype(np.int64) == gt_cls
                    iou_mask = iou[gi] >= args.match_iou
                    idx = np.where(cls_mask & iou_mask)[0]
                else:
                    idx = np.array([], dtype=np.int64)

                if len(idx):
                    order = idx[np.argsort(-d[idx, 4])]
                    top1 = float(d[order[0], 4])
                    top1_iou = float(iou[gi, order[0]])
                else:
                    order = np.array([], dtype=np.int64)
                    top1 = 0.0
                    top1_iou = 0.0

                if len(order) >= 2:
                    top2 = float(d[order[1], 4])
                    top2_iou = float(iou[gi, order[1]])
                else:
                    top2 = 0.0
                    top2_iou = 0.0

                rows.append({
                    "tag": args.tag,
                    "image": img_path.name,
                    "image_index": image_i,
                    "gt_index": gi,
                    "gt_class": gt_cls,
                    "candidate_count": int(len(order)),
                    "top1_score": top1,
                    "top2_score": top2,
                    "score_margin": top1 - top2,
                    "top1_iou": top1_iou,
                    "top2_iou": top2_iou,
                    "has_candidate": int(len(order) >= 1),
                    "has_competitor": int(len(order) >= 2),
                })

    detail = pd.DataFrame(rows)
    detail_path = out_dir / f"{args.tag}_score_margin_detail.csv"
    detail.to_csv(detail_path, index=False)

    if len(detail):
        summary = {
            "tag": args.tag,
            "num_gt": int(len(detail)),
            "gt_with_candidate": int(detail["has_candidate"].sum()),
            "gt_with_competitor": int(detail["has_competitor"].sum()),
            "candidate_count_mean": float(detail["candidate_count"].mean()),
            "candidate_count_median": float(detail["candidate_count"].median()),
            "top1_score_mean": float(detail["top1_score"].mean()),
            "top2_score_mean": float(detail["top2_score"].mean()),
            "score_margin_mean": float(detail["score_margin"].mean()),
            "score_margin_median": float(detail["score_margin"].median()),
            "top1_iou_mean": float(detail["top1_iou"].mean()),
            "top2_iou_mean": float(detail["top2_iou"].mean()),
            "match_iou_threshold": args.match_iou,
            "conf_threshold": args.conf,
            "max_det": args.max_det,
        }
    else:
        summary = {"tag": args.tag, "num_gt": 0}

    summary_df = pd.DataFrame([summary])
    summary_path = out_dir / f"{args.tag}_score_margin_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(summary_df.to_string(index=False))
    print("saved:", detail_path, summary_path)


if __name__ == "__main__":
    main()
