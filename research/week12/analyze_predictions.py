
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import inspect
import json
import sys
from typing import Any

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
from utils.plots import Annotator, colors

try:
    from utils.e2e_postprocess import (
        select_one2one,
        v10_no_nms_postprocess,
        class_aware_nms_postprocess,
    )
except Exception as e:
    select_one2one = None
    v10_no_nms_postprocess = None
    class_aware_nms_postprocess = None
    IMPORT_E2E_ERROR = repr(e)
else:
    IMPORT_E2E_ERROR = None


def load_names(data_yaml: Path):
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    return names


def load_image(path: Path, imgsz: int, stride: int = 32):
    im0 = cv2.imread(str(path))
    if im0 is None:
        raise FileNotFoundError(path)
    im = letterbox(im0, imgsz, stride=stride, auto=True)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).float() / 255.0
    if im.ndimension() == 3:
        im = im.unsqueeze(0)
    return im, im0


def as_det_tensor(obj: Any):
    if obj is None:
        return None
    if torch.is_tensor(obj):
        if obj.ndim == 3:
            return obj[0]
        return obj
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return torch.empty((0, 6))
        if torch.is_tensor(obj[0]):
            t = obj[0]
            if t.ndim == 3:
                return t[0]
            return t
        for x in obj:
            t = as_det_tensor(x)
            if t is not None:
                return t
    if isinstance(obj, dict):
        for key in ["pred", "preds", "detections", "det", "output"]:
            if key in obj:
                t = as_det_tensor(obj[key])
                if t is not None:
                    return t
    return None


def call_with_supported_kwargs(fn, *args, **kwargs):
    sig = inspect.signature(fn)
    usable = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(*args, **usable)


def e2e_postprocess(preds, mode: str, conf_thres: float, iou_thres: float, max_det: int):
    if select_one2one is None:
        raise RuntimeError(f"Could not import e2e_postprocess functions: {IMPORT_E2E_ERROR}")

    one2one = select_one2one(preds)

    if mode == "one2one_no_nms":
        out = call_with_supported_kwargs(
            v10_no_nms_postprocess,
            one2one,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
        )
    elif mode == "one2one_nms":
        out = call_with_supported_kwargs(
            class_aware_nms_postprocess,
            one2one,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
        )
    else:
        raise ValueError(mode)

    det = as_det_tensor(out)
    if det is None:
        raise RuntimeError(f"Could not convert e2e postprocess output to detection tensor. type={type(out)}")
    return det


def native_postprocess(preds, conf_thres: float, iou_thres: float, max_det: int):
    pred = preds[0] if isinstance(preds, (list, tuple)) else preds
    out = non_max_suppression(
        pred,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        classes=None,
        agnostic=False,
        max_det=max_det,
    )
    return out[0]


def scale_det(det: torch.Tensor, img_shape, im0_shape):
    if det is None:
        return torch.empty((0, 6))
    if det.numel() == 0:
        return det.reshape(0, 6).detach().cpu()
    det = det.detach().clone()
    if det.shape[1] < 6:
        raise RuntimeError(f"Detection tensor must have >=6 columns, got shape={tuple(det.shape)}")
    det[:, :4] = scale_boxes(img_shape, det[:, :4], im0_shape).round()
    return det[:, :6].detach().cpu()


def load_gt(label_path: Path, im0_shape):
    h, w = im0_shape[:2]
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, cx, cy, bw, bh = line.split()[:5]
        cls = int(cls)
        cx, cy, bw, bh = map(float, [cx, cy, bw, bh])
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        boxes.append([x1, y1, x2, y2, 1.0, cls])
    return boxes


def draw_boxes(im0, det, names, out_path: Path, title: str = ""):
    img = im0.copy()
    annotator = Annotator(img, line_width=2, example=str(names))
    if det is not None and len(det):
        for *xyxy, conf, cls in det.tolist():
            c = int(cls)
            label = f"{names[c] if c < len(names) else c} {conf:.2f}"
            annotator.box_label(xyxy, label, color=colors(c, True))
    result = annotator.result()
    if title:
        cv2.putText(result, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), result)


def draw_gt(im0, gt, names, out_path: Path):
    img = im0.copy()
    annotator = Annotator(img, line_width=2, example=str(names))
    for *xyxy, conf, cls in gt:
        c = int(cls)
        label = f"GT:{names[c] if c < len(names) else c}"
        annotator.box_label(xyxy, label, color=(0, 255, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotator.result())


def summarize_det(det: torch.Tensor, label: str, image_name: str):
    if det is None:
        det = torch.empty((0, 6))
    n = int(len(det))
    if n:
        conf = det[:, 4].float().numpy()
        xyxy = det[:, :4].float().numpy()
        width = np.maximum(0, xyxy[:, 2] - xyxy[:, 0])
        height = np.maximum(0, xyxy[:, 3] - xyxy[:, 1])
        area = width * height
        return {
            "label": label,
            "image": image_name,
            "num_pred": n,
            "conf_min": float(conf.min()),
            "conf_mean": float(conf.mean()),
            "conf_median": float(np.median(conf)),
            "conf_max": float(conf.max()),
            "area_mean": float(area.mean()),
            "area_median": float(np.median(area)),
        }
    return {
        "label": label,
        "image": image_name,
        "num_pred": 0,
        "conf_min": None,
        "conf_mean": None,
        "conf_median": None,
        "conf_max": None,
        "area_mean": None,
        "area_median": None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-weights", required=True)
    ap.add_argument("--proposed-weights", required=True)
    ap.add_argument("--sample-root", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    names = load_names(Path(args.data))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_root = Path(args.sample_root)
    images = sorted((sample_root / "images").glob("*.jpg"))
    assert images, f"no images in {sample_root / 'images'}"

    baseline = attempt_load(args.baseline_weights, device=device)
    proposed = attempt_load(args.proposed_weights, device=device)
    baseline.eval()
    proposed.eval()

    stride = 32
    if hasattr(baseline, "stride"):
        try:
            stride = int(baseline.stride.max())
        except Exception:
            stride = 32

    rows = []
    errors = []
    det_dump = {}

    for img_path in images:
        label_path = sample_root / "labels" / f"{img_path.stem}.txt"
        im, im0 = load_image(img_path, args.imgsz, stride=stride)
        im = im.to(device)

        gt = load_gt(label_path, im0.shape)
        draw_gt(im0, gt, names, out_dir / "gt_overlay" / img_path.name)

        modes = []

        with torch.no_grad():
            try:
                b_out = baseline(im)
                b_det = native_postprocess(b_out, args.conf, args.iou, args.max_det)
                b_det = scale_det(b_det, im.shape[2:], im0.shape)
                modes.append(("baseline_native_nms", b_det))
            except Exception as e:
                errors.append({"image": img_path.name, "label": "baseline_native_nms", "error": repr(e)})

            try:
                p_out = proposed(im)
                p_nat = native_postprocess(p_out, args.conf, args.iou, args.max_det)
                p_nat = scale_det(p_nat, im.shape[2:], im0.shape)
                modes.append(("proposed_native_nms", p_nat))
            except Exception as e:
                errors.append({"image": img_path.name, "label": "proposed_native_nms", "error": repr(e)})

            try:
                p_out = proposed(im)
                p_o2o_nms = e2e_postprocess(p_out, "one2one_nms", args.conf, args.iou, args.max_det)
                p_o2o_nms = scale_det(p_o2o_nms, im.shape[2:], im0.shape)
                modes.append(("proposed_one2one_nms", p_o2o_nms))
            except Exception as e:
                errors.append({"image": img_path.name, "label": "proposed_one2one_nms", "error": repr(e)})

            try:
                p_out = proposed(im)
                p_o2o_no = e2e_postprocess(p_out, "one2one_no_nms", args.conf, args.iou, args.max_det)
                p_o2o_no = scale_det(p_o2o_no, im.shape[2:], im0.shape)
                modes.append(("proposed_one2one_no_nms", p_o2o_no))
            except Exception as e:
                errors.append({"image": img_path.name, "label": "proposed_one2one_no_nms", "error": repr(e)})

        for label, det in modes:
            rows.append(summarize_det(det, label, img_path.name))
            draw_boxes(im0, det, names, out_dir / label / img_path.name, title=label)
            det_dump[f"{img_path.name}:{label}"] = det.tolist() if det is not None else []

    csv_path = out_dir / "prediction_count_confidence_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "label",
            "image",
            "num_pred",
            "conf_min",
            "conf_mean",
            "conf_median",
            "conf_max",
            "area_mean",
            "area_median",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "prediction_detections_raw.json").write_text(json.dumps(det_dump, indent=2), encoding="utf-8")
    (out_dir / "prediction_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")

    import pandas as pd
    df = pd.DataFrame(rows)
    agg = df.groupby("label").agg(
        images=("image", "count"),
        mean_num_pred=("num_pred", "mean"),
        median_num_pred=("num_pred", "median"),
        max_num_pred=("num_pred", "max"),
        mean_conf=("conf_mean", "mean"),
        median_conf=("conf_median", "median"),
        max_conf=("conf_max", "max"),
    ).reset_index()
    agg.to_csv(out_dir / "prediction_aggregate_summary.csv", index=False)

    print(agg)
    print("errors:", len(errors))
    if errors:
        print(errors[:5])
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
