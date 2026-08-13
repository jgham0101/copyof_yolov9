
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
from utils.general import non_max_suppression, scale_boxes

from utils.e2e_postprocess import (
    select_one2one,
    class_aware_nms_postprocess,
    v10_no_nms_postprocess,
)

from research.week20.official_v10_postprocess import (
    official_v10_two_stage_postprocess,
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

    for line in lab.read_text(encoding="utf-8").splitlines():
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


def normalize_det(obj):
    if obj is None:
        return torch.zeros((0, 6))

    if torch.is_tensor(obj):
        t = obj.detach()

        if t.ndim == 3:
            t = t[0]

        if t.ndim == 2 and t.shape[1] >= 6:
            return t[:, :6].float()

        if (
            t.ndim == 2
            and t.shape[0] >= 6
            and t.shape[1] > t.shape[0]
        ):
            return t[:6, :].T.float()

        return torch.zeros((0, 6))

    if isinstance(obj, (list, tuple)):
        if len(obj):
            return normalize_det(obj[0])

    if isinstance(obj, dict):
        for k in [
            "pred",
            "detections",
            "det",
            "output",
            "y",
        ]:
            if k in obj:
                out = normalize_det(obj[k])
                if len(out):
                    return out

    return torch.zeros((0, 6))


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


def summarize(det, gt):
    d = (
        det.detach().cpu().numpy()
        if torch.is_tensor(det)
        else np.asarray(det)
    )

    if d.ndim != 2 or d.shape[1] < 6:
        d = np.zeros((0, 6), dtype=np.float32)

    out = {
        "num_pred": int(len(d)),
        "mean_conf": None,
        "max_conf": None,
        "mean_best_iou": None,
        "match_iou50": 0,
        "class_match_iou50": 0,
        "duplicate_pairs_iou70_same_class": 0,
    }

    if len(d) == 0:
        return out

    out["mean_conf"] = float(d[:, 4].mean())
    out["max_conf"] = float(d[:, 4].max())

    if len(gt):
        iou = box_iou_np(
            d[:, :4],
            gt[:, :4],
        )

        best_gt = iou.argmax(axis=1)
        best_iou = iou.max(axis=1)

        pred_cls = d[:, 5].astype(np.int64)
        gt_cls = gt[
            best_gt,
            4,
        ].astype(np.int64)

        out["mean_best_iou"] = float(
            best_iou.mean()
        )

        out["match_iou50"] = int(
            (best_iou >= 0.5).sum()
        )

        out["class_match_iou50"] = int(
            (
                (best_iou >= 0.5)
                & (pred_cls == gt_cls)
            ).sum()
        )

    if len(d) > 1:
        pred_iou = box_iou_np(
            d[:, :4],
            d[:, :4],
        )

        cls = d[:, 5].astype(np.int64)
        dup = 0

        for i in range(len(d)):
            for j in range(i + 1, len(d)):
                if (
                    cls[i] == cls[j]
                    and pred_iou[i, j] >= 0.7
                ):
                    dup += 1

        out[
            "duplicate_pairs_iou70_same_class"
        ] = int(dup)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf-list", required=True)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--sample-n", type=int, default=128)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    confs = [
        float(x)
        for x in args.conf_list.split(",")
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

            raw = model(im)
            one = select_one2one(raw)

            for conf in confs:
                modes = {}

                pred_native = (
                    raw[0]
                    if isinstance(raw, (list, tuple))
                    else raw
                )

                modes["native_nms"] = normalize_det(
                    non_max_suppression(
                        pred_native,
                        conf_thres=conf,
                        iou_thres=args.iou,
                        max_det=args.max_det,
                    )[0]
                )

                modes["one2one_nms"] = normalize_det(
                    class_aware_nms_postprocess(
                        one,
                        conf_thres=conf,
                        iou_thres=args.iou,
                        max_det=args.max_det,
                    )
                )

                modes[
                    "one2one_current_no_nms"
                ] = normalize_det(
                    v10_no_nms_postprocess(
                        one,
                        conf_thres=conf,
                        max_det=args.max_det,
                    )
                )

                modes[
                    "one2one_official_v10"
                ] = normalize_det(
                    official_v10_two_stage_postprocess(
                        one,
                        conf_thres=conf,
                        max_det=args.max_det,
                    )
                )

                for mode, det in modes.items():
                    if len(det):
                        det = det.clone()
                        det[:, :4] = scale_boxes(
                            im.shape[2:],
                            det[:, :4],
                            im0.shape,
                        )

                    row = {
                        "tag": args.tag,
                        "image": img_path.name,
                        "image_index": image_i,
                        "mode": mode,
                        "conf_thres": conf,
                        "max_det": args.max_det,
                    }

                    row.update(
                        summarize(
                            det,
                            gt,
                        )
                    )

                    rows.append(row)

    detail = pd.DataFrame(rows)

    detail_path = (
        out_dir /
        f"{args.tag}_four_mode_detail.csv"
    )
    detail.to_csv(detail_path, index=False)

    agg = detail.groupby(
        [
            "tag",
            "mode",
            "conf_thres",
            "max_det",
        ],
        dropna=False,
    ).agg(
        images=("image", "count"),
        mean_num_pred=("num_pred", "mean"),
        median_num_pred=("num_pred", "median"),
        max_num_pred=("num_pred", "max"),
        mean_conf=("mean_conf", "mean"),
        max_conf=("max_conf", "max"),
        mean_best_iou=("mean_best_iou", "mean"),
        total_match_iou50=("match_iou50", "sum"),
        total_class_match_iou50=(
            "class_match_iou50",
            "sum",
        ),
        total_duplicate_pairs_iou70_same_class=(
            "duplicate_pairs_iou70_same_class",
            "sum",
        ),
    ).reset_index()

    agg_path = (
        out_dir /
        f"{args.tag}_four_mode_aggregate.csv"
    )
    agg.to_csv(agg_path, index=False)

    print(agg.to_string(index=False))
    print("saved:", detail_path)
    print("saved:", agg_path)


if __name__ == "__main__":
    main()
