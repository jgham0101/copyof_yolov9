from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def tload(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location="cpu")


def unwrap(ckpt):
    if isinstance(ckpt, dict):
        return ckpt.get("ema") or ckpt.get("model")
    return ckpt


def letterbox(im, new_shape=640, color=(114, 114, 114)):
    shape = im.shape[:2]

    r = min(
        new_shape / shape[0],
        new_shape / shape[1],
    )

    new_unpad = (
        int(round(shape[1] * r)),
        int(round(shape[0] * r)),
    )

    dw = (new_shape - new_unpad[0]) / 2
    dh = (new_shape - new_unpad[1]) / 2

    resized = np.array(
        Image.fromarray(im).resize(
            new_unpad,
            Image.BILINEAR,
        )
    )

    canvas = np.full(
        (new_shape, new_shape, 3),
        color,
        dtype=np.uint8,
    )

    x0 = int(round(dw - 0.1))
    y0 = int(round(dh - 0.1))

    canvas[
        y0:y0 + new_unpad[1],
        x0:x0 + new_unpad[0],
    ] = resized

    return canvas, r, (dw, dh)


def xywhn_to_xyxy(line, w0, h0, r, pad):
    c, x, y, w, h = map(float, line.split())

    x *= w0
    y *= h0
    w *= w0
    h *= h0

    return int(c), [
        (x - w / 2) * r + pad[0],
        (y - h / 2) * r + pad[1],
        (x + w / 2) * r + pad[0],
        (y + h / 2) * r + pad[1],
    ]


def iou_matrix(gt_boxes, boxes):
    # gt_boxes: [G,4], boxes: [N,4]
    lt = torch.maximum(
        gt_boxes[:, None, :2],
        boxes[None, :, :2],
    )
    rb = torch.minimum(
        gt_boxes[:, None, 2:],
        boxes[None, :, 2:],
    )

    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    ga = (
        (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0)
        *
        (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=0)
    )

    ba = (
        (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
        *
        (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    )

    return inter / (
        ga[:, None] + ba[None, :] - inter + 1e-9
    )


def extract_pred(model, x):
    out = model(x)

    if isinstance(out, tuple):
        pred = out[0]
    elif torch.is_tensor(out):
        pred = out
    else:
        raise TypeError(f"Unsupported eval output type: {type(out)}")

    if pred.ndim != 3:
        raise RuntimeError(f"Unexpected prediction shape: {pred.shape}")

    if pred.shape[1] >= 84 and pred.shape[2] > pred.shape[1]:
        pred = pred.permute(0, 2, 1)

    if pred.shape[-1] < 84:
        raise RuntimeError(f"Expected >=84 channels, got {pred.shape}")

    return pred


def safe_logit(p):
    p = torch.clamp(p, 1e-7, 1.0 - 1e-7)
    return torch.log(p) - torch.log1p(-p)


def quantiles(x):
    x = np.asarray(x, dtype=float)

    if x.size == 0:
        return {
            "mean": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
        }

    return {
        "mean": float(x.mean()),
        "p10": float(np.quantile(x, 0.10)),
        "p25": float(np.quantile(x, 0.25)),
        "median": float(np.quantile(x, 0.50)),
        "p75": float(np.quantile(x, 0.75)),
        "p90": float(np.quantile(x, 0.90)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--val-txt", required=True)
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")

    model = unwrap(tload(a.weights))
    if model is None:
        raise RuntimeError("Checkpoint has neither EMA nor model.")

    model = model.float().to(device).eval()

    image_paths = [
        x.strip()
        for x in Path(a.val_txt).read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ][:a.limit]

    labels_root = Path(a.labels_root)

    acc = defaultdict(list)
    total_gt = 0
    gt_with_iou50_pos = 0

    with torch.no_grad():
        for ip in tqdm(image_paths, desc="week31-cls-probe"):
            p = Path(ip)

            if not p.is_absolute():
                p = (
                    Path("/content/datasets/coco")
                    / ip.replace("./", "")
                )

            if not p.exists():
                raise FileNotFoundError(p)

            label_path = labels_root / f"{p.stem}.txt"
            if not label_path.exists():
                continue

            lines = [
                z.strip()
                for z in label_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if z.strip()
            ]

            if not lines:
                continue

            im0 = np.array(Image.open(p).convert("RGB"))
            h0, w0 = im0.shape[:2]
            im, r, pad = letterbox(im0, 640)

            gt_classes = []
            gt_boxes = []

            for line in lines:
                c, box = xywhn_to_xyxy(
                    line,
                    w0,
                    h0,
                    r,
                    pad,
                )
                gt_classes.append(c)
                gt_boxes.append(box)

            gt_classes = torch.tensor(
                gt_classes,
                device=device,
                dtype=torch.long,
            )

            gt_boxes = torch.tensor(
                gt_boxes,
                device=device,
                dtype=torch.float32,
            )

            x = (
                torch.from_numpy(im)
                .permute(2, 0, 1)
                .contiguous()
                .unsqueeze(0)
                .to(device)
                .float()
                / 255.0
            )

            pred = extract_pred(model, x)[0]

            xywh = pred[:, :4]
            cx, cy, bw, bh = xywh.unbind(1)

            boxes = torch.stack(
                [
                    cx - bw / 2,
                    cy - bh / 2,
                    cx + bw / 2,
                    cy + bh / 2,
                ],
                dim=1,
            )

            scores = pred[:, 4:84]

            if (
                float(scores.min()) < 0.0
                or float(scores.max()) > 1.0
            ):
                scores = scores.sigmoid()

            logits = safe_logit(scores)

            ious = iou_matrix(gt_boxes, boxes)

            class_to_gt = defaultdict(list)
            for gi, c in enumerate(gt_classes.tolist()):
                class_to_gt[int(c)].append(gi)

            for gi in range(gt_boxes.shape[0]):
                c = int(gt_classes[gi].item())
                iou = ious[gi]

                spatial_idx = int(torch.argmax(iou).item())
                spatial_iou = float(iou[spatial_idx].item())

                gt_score = scores[spatial_idx, c]
                gt_logit = logits[spatial_idx, c]

                row_scores = scores[spatial_idx]
                row_logits = logits[spatial_idx]

                wrong_scores = torch.cat(
                    [row_scores[:c], row_scores[c+1:]]
                )
                wrong_logits = torch.cat(
                    [row_logits[:c], row_logits[c+1:]]
                )

                wrong_score = wrong_scores.max()
                wrong_logit = wrong_logits.max()

                class_rank = int(
                    1 + (row_scores > gt_score).sum().item()
                )

                acc["spatial_iou"].append(spatial_iou)
                acc["spatial_gt_score"].append(float(gt_score.item()))
                acc["spatial_gt_logit"].append(float(gt_logit.item()))
                acc["spatial_wrongmax_score"].append(
                    float(wrong_score.item())
                )
                acc["spatial_wrongmax_logit"].append(
                    float(wrong_logit.item())
                )
                acc["spatial_class_margin_score"].append(
                    float((gt_score - wrong_score).item())
                )
                acc["spatial_class_margin_logit"].append(
                    float((gt_logit - wrong_logit).item())
                )
                acc["spatial_gt_rank"].append(class_rank)
                acc["spatial_rank1"].append(int(class_rank == 1))
                acc["spatial_rank5"].append(int(class_rank <= 5))

                pos_mask = iou >= 0.50

                if pos_mask.any():
                    gt_with_iou50_pos += 1

                    pos_scores = scores[pos_mask, c]
                    pos_logits = logits[pos_mask, c]

                    pos_max_score = pos_scores.max()
                    pos_max_logit = pos_logits.max()

                    acc["positive_max_score"].append(
                        float(pos_max_score.item())
                    )
                    acc["positive_max_logit"].append(
                        float(pos_max_logit.item())
                    )

                    same_cls_gt_indices = class_to_gt[c]

                    same_cls_max_iou = ious[
                        same_cls_gt_indices
                    ].max(dim=0).values

                    hard_bg_mask = same_cls_max_iou < 0.10

                    if hard_bg_mask.any():
                        hard_bg_scores = scores[hard_bg_mask, c]
                        hard_bg_logits = logits[hard_bg_mask, c]

                        hard_bg_max_score = hard_bg_scores.max()
                        hard_bg_max_logit = hard_bg_logits.max()

                        acc["hard_bg_max_score"].append(
                            float(hard_bg_max_score.item())
                        )
                        acc["hard_bg_max_logit"].append(
                            float(hard_bg_max_logit.item())
                        )
                        acc["pos_minus_bg_score"].append(
                            float(
                                (pos_max_score - hard_bg_max_score).item()
                            )
                        )
                        acc["pos_minus_bg_logit"].append(
                            float(
                                (pos_max_logit - hard_bg_max_logit).item()
                            )
                        )
                        acc["pos_wins_hard_bg"].append(
                            int(pos_max_score > hard_bg_max_score)
                        )

                total_gt += 1

    report = {
        "images": len(image_paths),
        "gt": total_gt,
        "gt_with_iou50_positive_rate": (
            gt_with_iou50_pos / total_gt
            if total_gt else None
        ),
        "spatial_gt_score": quantiles(acc["spatial_gt_score"]),
        "spatial_gt_logit": quantiles(acc["spatial_gt_logit"]),
        "spatial_wrongmax_score": quantiles(acc["spatial_wrongmax_score"]),
        "spatial_wrongmax_logit": quantiles(acc["spatial_wrongmax_logit"]),
        "spatial_class_margin_score":
            quantiles(acc["spatial_class_margin_score"]),
        "spatial_class_margin_logit":
            quantiles(acc["spatial_class_margin_logit"]),
        "spatial_rank1_rate":
            float(np.mean(acc["spatial_rank1"])),
        "spatial_rank5_rate":
            float(np.mean(acc["spatial_rank5"])),
        "spatial_gt_rank_median":
            float(np.median(acc["spatial_gt_rank"])),
        "positive_max_score":
            quantiles(acc["positive_max_score"]),
        "positive_max_logit":
            quantiles(acc["positive_max_logit"]),
        "hard_bg_max_score":
            quantiles(acc["hard_bg_max_score"]),
        "hard_bg_max_logit":
            quantiles(acc["hard_bg_max_logit"]),
        "pos_minus_bg_score":
            quantiles(acc["pos_minus_bg_score"]),
        "pos_minus_bg_logit":
            quantiles(acc["pos_minus_bg_logit"]),
        "pos_wins_hard_bg_rate":
            float(np.mean(acc["pos_wins_hard_bg"])),
    }

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
