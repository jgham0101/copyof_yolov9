from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ============================================================
# IMPORTANT:
# This script lives under:
#   /content/copyof_yolov9/research/
#
# YOLOv9 checkpoints contain pickled classes such as
#   models.yolo.DetectionModel
#
# Therefore the repository ROOT must be importable BEFORE
# torch.load() is called.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


def tload(path):

    try:

        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        return torch.load(
            path,
            map_location="cpu",
        )


def unwrap(ckpt):

    if isinstance(
        ckpt,
        dict,
    ):

        return (
            ckpt.get("ema")
            or ckpt.get("model")
        )

    return ckpt


def letterbox(
    im,
    new_shape=640,
    color=(114, 114, 114),
):

    shape = im.shape[:2]

    r = min(
        new_shape / shape[0],
        new_shape / shape[1],
    )

    new_unpad = (
        int(
            round(
                shape[1] * r
            )
        ),
        int(
            round(
                shape[0] * r
            )
        ),
    )

    dw = (
        new_shape
        - new_unpad[0]
    )

    dh = (
        new_shape
        - new_unpad[1]
    )

    dw /= 2
    dh /= 2


    resized = np.array(

        Image.fromarray(
            im
        ).resize(
            new_unpad,
            Image.BILINEAR,
        )
    )


    canvas = np.full(
        (
            new_shape,
            new_shape,
            3,
        ),
        color,
        dtype=np.uint8,
    )


    x0 = int(
        round(
            dw - 0.1
        )
    )

    y0 = int(
        round(
            dh - 0.1
        )
    )


    canvas[
        y0:y0 + new_unpad[1],
        x0:x0 + new_unpad[0],
    ] = resized


    return (
        canvas,
        r,
        (
            dw,
            dh,
        ),
    )


def xywhn_to_xyxy(
    line,
    w0,
    h0,
    r,
    pad,
):

    c, x, y, w, h = map(
        float,
        line.split(),
    )


    x *= w0
    y *= h0
    w *= w0
    h *= h0


    return (
        int(c),
        [
            (x - w / 2) * r + pad[0],
            (y - h / 2) * r + pad[1],
            (x + w / 2) * r + pad[0],
            (y + h / 2) * r + pad[1],
        ],
    )


def box_iou_one_to_many(
    gt,
    boxes,
):

    x1 = torch.maximum(
        gt[0],
        boxes[:, 0],
    )

    y1 = torch.maximum(
        gt[1],
        boxes[:, 1],
    )

    x2 = torch.minimum(
        gt[2],
        boxes[:, 2],
    )

    y2 = torch.minimum(
        gt[3],
        boxes[:, 3],
    )


    inter = (
        (x2 - x1).clamp(0)
        *
        (y2 - y1).clamp(0)
    )


    ga = (
        (gt[2] - gt[0]).clamp(0)
        *
        (gt[3] - gt[1]).clamp(0)
    )


    ba = (
        (boxes[:, 2] - boxes[:, 0]).clamp(0)
        *
        (boxes[:, 3] - boxes[:, 1]).clamp(0)
    )


    return (
        inter
        /
        (
            ga
            + ba
            - inter
            + 1e-9
        )
    )


def extract_pred(
    model,
    x,
):

    out = model(
        x
    )


    if isinstance(
        out,
        tuple,
    ):

        pred = out[0]


    elif torch.is_tensor(
        out
    ):

        pred = out


    else:

        raise TypeError(
            "Unsupported eval output type: "
            f"{type(out)}"
        )


    if pred.ndim != 3:

        raise RuntimeError(
            "Unexpected prediction shape: "
            f"{pred.shape}"
        )


    # DDetect commonly returns:
    #
    # [B, 4+nc, N]
    #
    # Convert to:
    #
    # [B, N, 4+nc]
    #
    if (
        pred.shape[1] >= 84
        and
        pred.shape[2]
        > pred.shape[1]
    ):

        pred = pred.permute(
            0,
            2,
            1,
        )


    if pred.shape[-1] < 84:

        raise RuntimeError(
            "Expected >=84 output channels, "
            f"got {pred.shape}"
        )


    return pred


def main():

    ap = argparse.ArgumentParser()


    ap.add_argument(
        "--weights",
        required=True,
    )

    ap.add_argument(
        "--val-txt",
        required=True,
    )

    ap.add_argument(
        "--labels-root",
        required=True,
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=500,
    )

    ap.add_argument(
        "--out",
        required=True,
    )


    a = ap.parse_args()


    if not torch.cuda.is_available():

        raise RuntimeError(
            "Week30 candidate probe requires GPU."
        )


    device = torch.device(
        "cuda:0"
    )


    # ========================================================
    # Load model
    # ========================================================

    model = unwrap(
        tload(
            a.weights
        )
    )


    if model is None:

        raise RuntimeError(
            "Checkpoint contains neither EMA nor model."
        )


    model = (
        model
        .float()
        .to(device)
        .eval()
    )


    # ========================================================
    # Validation list
    # ========================================================

    image_paths = [

        x.strip()

        for x in Path(
            a.val_txt
        ).read_text(
            encoding="utf-8"
        ).splitlines()

        if x.strip()

    ][:a.limit]


    if not image_paths:

        raise RuntimeError(
            "No validation images found."
        )


    labels_root = Path(
        a.labels_root
    )


    best_ious = []

    correct_best_ious = []

    spatial_gt_scores = []

    spatial_class_correct = []

    total_gt = 0


    # ========================================================
    # Probe
    # ========================================================

    with torch.no_grad():

        for ip in tqdm(
            image_paths,
            desc="candidate-probe",
        ):

            p = Path(
                ip
            )


            if not p.is_absolute():

                p = (
                    Path(
                        "/content/datasets/coco"
                    )
                    /
                    ip.replace(
                        "./",
                        "",
                    )
                )


            if not p.exists():

                raise FileNotFoundError(
                    p
                )


            im0 = np.array(

                Image.open(
                    p
                ).convert(
                    "RGB"
                )
            )


            h0, w0 = (
                im0.shape[:2]
            )


            im, r, pad = letterbox(
                im0,
                640,
            )


            x = (
                torch.from_numpy(
                    im
                )
                .permute(
                    2,
                    0,
                    1,
                )
                .contiguous()
            )


            x = (
                x.unsqueeze(0)
                .to(device)
                .float()
                / 255.0
            )


            pred = extract_pred(
                model,
                x,
            )[0]


            boxes = pred[
                :,
                :4,
            ]


            cx, cy, bw, bh = (
                boxes.unbind(
                    1
                )
            )


            boxes_xyxy = torch.stack(

                [
                    cx - bw / 2,
                    cy - bh / 2,
                    cx + bw / 2,
                    cy + bh / 2,
                ],

                dim=1,
            )


            scores = pred[
                :,
                4:,
            ]


            # Native decoded output should already contain
            # probabilities. Keep a defensive fallback only.
            if (
                float(
                    scores.min()
                ) < 0.0
                or
                float(
                    scores.max()
                ) > 1.0
            ):

                scores = scores.sigmoid()


            label_path = (
                labels_root
                /
                f"{p.stem}.txt"
            )


            if not label_path.exists():

                continue


            lines = [

                z.strip()

                for z in label_path.read_text(
                    encoding="utf-8"
                ).splitlines()

                if z.strip()
            ]


            for line in lines:

                cls, gt_list = (
                    xywhn_to_xyxy(
                        line,
                        w0,
                        h0,
                        r,
                        pad,
                    )
                )


                gt = torch.tensor(
                    gt_list,
                    device=device,
                    dtype=torch.float32,
                )


                ious = (
                    box_iou_one_to_many(
                        gt,
                        boxes_xyxy,
                    )
                )


                best_idx = int(
                    torch.argmax(
                        ious
                    ).item()
                )


                best_iou = float(
                    ious[
                        best_idx
                    ].item()
                )


                gt_scores = scores[
                    :,
                    cls,
                ]


                top_classes = (
                    scores.argmax(
                        1
                    )
                )


                correct_mask = (
                    top_classes
                    == cls
                )


                if correct_mask.any():

                    correct_best = float(

                        ious[
                            correct_mask
                        ].max().item()
                    )

                else:

                    correct_best = 0.0


                best_ious.append(
                    best_iou
                )


                correct_best_ious.append(
                    correct_best
                )


                spatial_gt_scores.append(

                    float(
                        gt_scores[
                            best_idx
                        ].item()
                    )
                )


                spatial_class_correct.append(

                    int(
                        top_classes[
                            best_idx
                        ].item()
                        == cls
                    )
                )


                total_gt += 1


    if total_gt == 0:

        raise RuntimeError(
            "Candidate probe found zero GT objects."
        )


    # ========================================================
    # Aggregate
    # ========================================================

    best = np.asarray(
        best_ious,
        dtype=float,
    )


    corr = np.asarray(
        correct_best_ious,
        dtype=float,
    )


    score = np.asarray(
        spatial_gt_scores,
        dtype=float,
    )


    classok = np.asarray(
        spatial_class_correct,
        dtype=float,
    )


    report = {

        "images":
            len(
                image_paths
            ),

        "gt":
            total_gt,


        "best_spatial_iou_mean":
            float(
                best.mean()
            ),

        "best_spatial_iou_median":
            float(
                np.median(
                    best
                )
            ),

        "best_spatial_iou50_rate":
            float(
                (
                    best >= 0.50
                ).mean()
            ),

        "best_spatial_iou75_rate":
            float(
                (
                    best >= 0.75
                ).mean()
            ),


        "correct_class_best_iou_mean":
            float(
                corr.mean()
            ),

        "correct_class_iou50_rate":
            float(
                (
                    corr >= 0.50
                ).mean()
            ),

        "correct_class_iou75_rate":
            float(
                (
                    corr >= 0.75
                ).mean()
            ),


        "spatial_best_gt_class_score_mean":
            float(
                score.mean()
            ),

        "spatial_best_gt_class_score_median":
            float(
                np.median(
                    score
                )
            ),

        "spatial_best_topclass_correct_rate":
            float(
                classok.mean()
            ),
    }


    Path(
        a.out
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    Path(
        a.out
    ).write_text(

        json.dumps(
            report,
            indent=2,
        ),

        encoding="utf-8",
    )


    print(
        json.dumps(
            report,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
