
import argparse
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
from utils.e2e_postprocess import (
    _as_bnc,
    select_one2one,
    xywh2xyxy,
)


def image_paths_from_data(data_yaml, sample_n):
    import yaml
    data = yaml.safe_load(
        Path(data_yaml).read_text(encoding="utf-8")
    )
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

    for line in lab.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        cls, xc, yc, bw, bh = [
            float(x) for x in line.split()[:5]
        ]

        rows.append([
            (xc - bw / 2) * w,
            (yc - bh / 2) * h,
            (xc + bw / 2) * w,
            (yc + bh / 2) * h,
            cls,
        ])

    return np.asarray(rows, dtype=np.float32)


def load_img(path, imgsz, stride):
    im0 = cv2.imread(str(path))
    if im0 is None:
        raise FileNotFoundError(path)

    im = letterbox(
        im0,
        imgsz,
        stride=stride,
        auto=True,
    )[0]

    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).float() / 255.0

    if im.ndim == 3:
        im = im.unsqueeze(0)

    return im, im0


def box_iou_np(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros(
            (len(a), len(b)),
            dtype=np.float32,
        )

    tl = np.maximum(
        a[:, None, :2],
        b[None, :, :2],
    )
    br = np.minimum(
        a[:, None, 2:4],
        b[None, :, 2:4],
    )
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]

    aa = (
        np.clip(a[:, 2] - a[:, 0], 0, None)
        * np.clip(a[:, 3] - a[:, 1], 0, None)
    )
    bb = (
        np.clip(b[:, 2] - b[:, 0], 0, None)
        * np.clip(b[:, 3] - b[:, 1], 0, None)
    )

    return (
        inter /
        (aa[:, None] + bb[None, :] - inter + 1e-9)
    )


def top2_margin(values):
    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0:
        return 0.0, 0.0, 0.0

    order = np.sort(values)[::-1]
    top1 = float(order[0])
    top2 = float(order[1]) if len(order) > 1 else 0.0

    return top1, top2, top1 - top2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--sample-n", type=int, default=128)
    ap.add_argument("--iou-thresholds", default="0.3,0.5")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [
        float(x)
        for x in args.iou_thresholds.split(",")
        if x.strip()
    ]

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    model = attempt_load(
        args.weights,
        device=device,
    )
    model.eval()

    stride = 32
    if hasattr(model, "stride"):
        try:
            stride = int(model.stride.max())
        except Exception:
            pass

    imgs = image_paths_from_data(
        args.data,
        args.sample_n,
    )

    assert imgs, "no images found"

    rows = []

    with torch.no_grad():
        for image_i, img_path in enumerate(imgs):
            im, im0 = load_img(
                img_path,
                args.imgsz,
                stride,
            )
            im = im.to(device)

            gt = load_gt(
                img_path,
                im0.shape,
            )

            if len(gt) == 0:
                continue

            raw = model(im)

            pred = _as_bnc(
                select_one2one(raw)
            )[0]

            boxes = xywh2xyxy(
                pred[:, :4].clone()
            )

            boxes = scale_boxes(
                im.shape[2:],
                boxes,
                im0.shape,
            )

            cls_scores = pred[:, 4:]
            max_scores, pred_labels = cls_scores.max(dim=1)

            boxes_np = boxes.detach().cpu().numpy().astype(np.float32)
            max_scores_np = max_scores.detach().cpu().numpy().astype(np.float32)
            pred_labels_np = pred_labels.detach().cpu().numpy().astype(np.int64)
            cls_scores_np = cls_scores.detach().cpu().numpy().astype(np.float32)

            iou = box_iou_np(
                gt[:, :4],
                boxes_np,
            )

            for gi, g in enumerate(gt):
                gt_cls = int(g[4])

                for thr in thresholds:
                    spatial_idx = np.where(
                        iou[gi] >= thr
                    )[0]

                    spatial_scores = (
                        max_scores_np[spatial_idx]
                        if len(spatial_idx)
                        else np.asarray([], dtype=np.float32)
                    )

                    s1, s2, smargin = top2_margin(
                        spatial_scores
                    )

                    spatial_top_class_correct = 0
                    spatial_top_iou = 0.0

                    if len(spatial_idx):
                        local_best_pos = int(
                            np.argmax(spatial_scores)
                        )
                        best_anchor = int(
                            spatial_idx[local_best_pos]
                        )
                        spatial_top_class_correct = int(
                            pred_labels_np[best_anchor] == gt_cls
                        )
                        spatial_top_iou = float(
                            iou[gi, best_anchor]
                        )

                    if len(spatial_idx):
                        class_idx = spatial_idx[
                            pred_labels_np[spatial_idx] == gt_cls
                        ]
                    else:
                        class_idx = np.asarray([], dtype=np.int64)

                    class_scores = (
                        max_scores_np[class_idx]
                        if len(class_idx)
                        else np.asarray([], dtype=np.float32)
                    )

                    c1, c2, cmargin = top2_margin(
                        class_scores
                    )

                    gt_class_scores = (
                        cls_scores_np[spatial_idx, gt_cls]
                        if len(spatial_idx)
                        else np.asarray([], dtype=np.float32)
                    )

                    g1, g2, gmargin = top2_margin(
                        gt_class_scores
                    )

                    gt_score_top_iou = 0.0

                    if len(spatial_idx):
                        best_gt_score_pos = int(
                            np.argmax(gt_class_scores)
                        )
                        best_gt_anchor = int(
                            spatial_idx[best_gt_score_pos]
                        )
                        gt_score_top_iou = float(
                            iou[gi, best_gt_anchor]
                        )

                    rows.append({
                        "tag": args.tag,
                        "image": img_path.name,
                        "image_index": image_i,
                        "gt_index": gi,
                        "gt_class": gt_cls,
                        "iou_threshold": thr,

                        "spatial_candidate_count":
                            int(len(spatial_idx)),

                        "spatial_top1_score": s1,
                        "spatial_top2_score": s2,
                        "spatial_score_margin": smargin,

                        "spatial_top_class_correct":
                            spatial_top_class_correct,

                        "spatial_top_iou":
                            spatial_top_iou,

                        "correct_pred_class_candidate_count":
                            int(len(class_idx)),

                        "classaware_top1_score": c1,
                        "classaware_top2_score": c2,
                        "classaware_score_margin": cmargin,

                        "gtclass_top1_score": g1,
                        "gtclass_top2_score": g2,
                        "gtclass_score_margin": gmargin,

                        "gtclass_top_anchor_iou":
                            gt_score_top_iou,

                        "has_spatial_candidate":
                            int(len(spatial_idx) > 0),

                        "has_correct_pred_class_candidate":
                            int(len(class_idx) > 0),
                    })

    detail = pd.DataFrame(rows)

    detail_path = (
        out_dir /
        f"{args.tag}_refined_ranking_detail.csv"
    )
    detail.to_csv(detail_path, index=False)

    summary_rows = []

    for thr in thresholds:
        d = detail[
            np.isclose(
                detail["iou_threshold"],
                thr,
            )
        ].copy()

        spatial_cond = d[
            d["has_spatial_candidate"] == 1
        ]

        class_cond = d[
            d["has_correct_pred_class_candidate"] == 1
        ]

        summary_rows.append({
            "tag": args.tag,
            "iou_threshold": thr,
            "num_gt": int(len(d)),

            "gt_with_spatial_candidate":
                int(d["has_spatial_candidate"].sum()),

            "gt_with_correct_pred_class_candidate":
                int(d["has_correct_pred_class_candidate"].sum()),

            "spatial_candidate_count_mean":
                float(d["spatial_candidate_count"].mean()),

            "spatial_margin_all_gt_mean":
                float(d["spatial_score_margin"].mean()),

            "spatial_margin_cond_mean":
                float(spatial_cond["spatial_score_margin"].mean())
                if len(spatial_cond) else 0.0,

            "spatial_margin_cond_median":
                float(spatial_cond["spatial_score_margin"].median())
                if len(spatial_cond) else 0.0,

            "spatial_top_class_accuracy_cond":
                float(spatial_cond["spatial_top_class_correct"].mean())
                if len(spatial_cond) else 0.0,

            "classaware_margin_cond_mean":
                float(class_cond["classaware_score_margin"].mean())
                if len(class_cond) else 0.0,

            "classaware_margin_cond_median":
                float(class_cond["classaware_score_margin"].median())
                if len(class_cond) else 0.0,

            "gtclass_margin_cond_mean":
                float(spatial_cond["gtclass_score_margin"].mean())
                if len(spatial_cond) else 0.0,

            "gtclass_margin_cond_median":
                float(spatial_cond["gtclass_score_margin"].median())
                if len(spatial_cond) else 0.0,

            "gtclass_top_anchor_iou_cond_mean":
                float(spatial_cond["gtclass_top_anchor_iou"].mean())
                if len(spatial_cond) else 0.0,
        })

    summary = pd.DataFrame(summary_rows)

    summary_path = (
        out_dir /
        f"{args.tag}_refined_ranking_summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    print(summary.to_string(index=False))
    print("saved:", detail_path)
    print("saved:", summary_path)


if __name__ == "__main__":
    main()
