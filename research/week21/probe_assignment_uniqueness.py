
import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.experimental import attempt_load
from utils.dataloaders import create_dataloader
from utils.loss_tal_dual import ComputeLoss
from utils.metrics import bbox_iou
from utils.tal.anchor_generator import make_anchors
from utils.tal.assigner import (
    select_candidates_in_gts,
    select_highest_overlaps,
)


def resolve_dataset_path(data_yaml):
    data = yaml.safe_load(
        Path(data_yaml).read_text(encoding="utf-8")
    )

    root = Path(data.get("path", ""))
    val = data.get("val", data.get("train"))

    if isinstance(val, list):
        val = val[0]

    p = Path(val)

    if not p.is_absolute():
        p = root / p

    return str(p), data


def make_loader(path, imgsz, batch, stride, workers):
    sig = inspect.signature(create_dataloader)

    optional = {
        "single_cls": False,
        "hyp": None,
        "augment": False,
        "cache": False,
        "pad": 0.5,
        "rect": True,
        "rank": -1,
        "workers": workers,
        "image_weights": False,
        "quad": False,
        "prefix": "week21: ",
        "shuffle": False,
        "seed": 0,
    }

    kwargs = {
        k: v
        for k, v in optional.items()
        if k in sig.parameters
    }

    loader, dataset = create_dataloader(
        path,
        imgsz,
        batch,
        stride,
        **kwargs,
    )

    return loader, dataset


def extract_raw_o2o(output):
    # Expected eval contract:
    # ([decoded_o2m, decoded_o2o],
    #  {"one2many": raw_o2m, "one2one": raw_o2o})
    if isinstance(output, tuple):
        for item in output:
            if isinstance(item, dict) and "one2one" in item:
                return item["one2one"]

    if isinstance(output, dict):
        if "one2one" in output:
            return output["one2one"]

    if isinstance(output, (list, tuple)):
        for item in output:
            if isinstance(item, dict) and "one2one" in item:
                return item["one2one"]

    raise RuntimeError(
        "Could not extract raw one2one features from model output. "
        f"type={type(output)}"
    )


def standard_box_iou(gt_boxes, pred_boxes):
    # gt: Gx4, pred: Nx4, xyxy
    if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
        return gt_boxes.new_zeros(
            (gt_boxes.shape[0], pred_boxes.shape[0])
        )

    lt = torch.maximum(
        gt_boxes[:, None, :2],
        pred_boxes[None, :, :2],
    )

    rb = torch.minimum(
        gt_boxes[:, None, 2:],
        pred_boxes[None, :, 2:],
    )

    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    gt_area = (
        (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0)
        * (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=0)
    )

    pred_area = (
        (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0)
        * (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0)
    )

    return (
        inter
        / (
            gt_area[:, None]
            + pred_area[None, :]
            - inter
            + 1e-9
        )
    )


def top2(values, indices):
    if len(indices) == 0:
        return {
            "top1_idx": -1,
            "top2_idx": -1,
            "top1": 0.0,
            "top2": 0.0,
            "gap": 0.0,
            "rel_gap": 0.0,
        }

    vals = values[indices]

    k = min(2, int(vals.numel()))
    v, pos = torch.topk(
        vals,
        k=k,
        largest=True,
        sorted=True,
    )

    i1 = int(indices[int(pos[0])].item())
    v1 = float(v[0].item())

    if k >= 2:
        i2 = int(indices[int(pos[1])].item())
        v2 = float(v[1].item())
    else:
        i2 = -1
        v2 = 0.0

    gap = v1 - v2

    return {
        "top1_idx": i1,
        "top2_idx": i2,
        "top1": v1,
        "top2": v2,
        "gap": gap,
        "rel_gap": gap / (abs(v1) + 1e-12),
    }


def scale_name(level):
    if level < 0:
        return "NONE"
    return f"P{level + 3}"


def summarize_rows(detail):
    rows = []

    for tag, d in detail.groupby("tag"):
        def mean_bool(col, condition=None):
            x = d if condition is None else d[condition]
            if len(x) == 0:
                return 0.0
            return float(
                pd.to_numeric(
                    x[col],
                    errors="coerce",
                ).fillna(0).mean()
            )

        selected = d["selected_after_conflict"] == 1
        iou50 = d["iou50_candidate_count"] > 0
        selected_iou50 = selected & (
            pd.to_numeric(
                d["selected_std_iou"],
                errors="coerce",
            ).fillna(0) >= 0.5
        )

        summary = {
            "tag": tag,
            "num_gt": int(len(d)),

            "selected_before_rate":
                mean_bool("selected_before_conflict"),

            "selected_after_rate":
                mean_bool("selected_after_conflict"),

            "lost_by_conflict_rate":
                mean_bool("lost_by_conflict"),

            "preselected_anchor_multi_gt_rate":
                mean_bool("preselected_anchor_multi_gt"),

            "alignment_gap_mean":
                float(d["align_gap"].mean()),

            "alignment_gap_median":
                float(d["align_gap"].median()),

            "alignment_rel_gap_median":
                float(d["align_rel_gap"].median()),

            "selected_std_iou_mean":
                float(
                    d.loc[selected, "selected_std_iou"].mean()
                ) if selected.any() else 0.0,

            "selected_pred_class_correct_rate":
                mean_bool(
                    "selected_pred_class_correct",
                    selected,
                ),

            "selected_equals_best_std_iou_rate":
                mean_bool(
                    "selected_equals_best_std_iou",
                    selected,
                ),

            "selected_equals_best_gtclass_inside_rate":
                mean_bool(
                    "selected_equals_best_gtclass_inside",
                    selected,
                ),

            "iou50_gt_rate":
                float(iou50.mean()),

            "iou50_candidate_count_mean":
                float(d["iou50_candidate_count"].mean()),

            "iou50_candidate_count_median":
                float(d["iou50_candidate_count"].median()),

            "iou50_cross_scale_gt_rate":
                mean_bool(
                    "iou50_has_cross_scale_candidates",
                    iou50,
                ),

            "iou50_scale_count_mean":
                float(
                    d.loc[
                        iou50,
                        "iou50_scale_count",
                    ].mean()
                ) if iou50.any() else 0.0,

            "selected_matches_inference_maxscore_iou50_rate":
                mean_bool(
                    "selected_matches_inference_maxscore_iou50",
                    selected_iou50,
                ),

            "selected_matches_inference_gtclass_iou50_rate":
                mean_bool(
                    "selected_matches_inference_gtclass_iou50",
                    selected_iou50,
                ),

            "selected_gtclass_wins_iou50_rate":
                mean_bool(
                    "selected_gtclass_outscores_iou50_competitors",
                    selected_iou50,
                ),

            "selected_maxscore_wins_iou50_rate":
                mean_bool(
                    "selected_maxscore_outscores_iou50_competitors",
                    selected_iou50,
                ),

            "selected_minus_best_comp_gtclass_median":
                float(
                    d.loc[
                        selected_iou50,
                        "selected_minus_best_comp_gtclass",
                    ].median()
                ) if selected_iou50.any() else 0.0,

            "selected_minus_best_comp_maxscore_median":
                float(
                    d.loc[
                        selected_iou50,
                        "selected_minus_best_comp_maxscore",
                    ].median()
                ) if selected_iou50.any() else 0.0,

            "same_scale_iou50_competitors_mean":
                float(
                    d.loc[
                        selected_iou50,
                        "same_scale_iou50_competitors",
                    ].mean()
                ) if selected_iou50.any() else 0.0,

            "cross_scale_iou50_competitors_mean":
                float(
                    d.loc[
                        selected_iou50,
                        "cross_scale_iou50_competitors",
                    ].mean()
                ) if selected_iou50.any() else 0.0,
        }

        for level in range(3):
            summary[
                f"selected_{scale_name(level)}_rate"
            ] = float(
                (
                    d.loc[selected, "selected_level"]
                    == level
                ).mean()
            ) if selected.any() else 0.0

            summary[
                f"iou50_{scale_name(level)}_candidate_mean"
            ] = float(
                d[f"iou50_level{level}_count"].mean()
            )

        rows.append(summary)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-images", type=int, default=128)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = attempt_load(
        args.weights,
        device=device,
    )

    model.eval()

    criterion = ComputeLoss(model)

    assert int(criterion.assigner2.topk) == 1, (
        f"O2O assigner topk must be 1, "
        f"got {criterion.assigner2.topk}"
    )

    val_path, data_dict = resolve_dataset_path(
        args.data
    )

    stride = int(
        model.stride.max().item()
        if torch.is_tensor(model.stride)
        else max(model.stride)
    )

    loader, _ = make_loader(
        val_path,
        args.imgsz,
        args.batch,
        stride,
        args.workers,
    )

    rows = []
    image_counter = 0
    printed_contract = False

    with torch.no_grad():

        for batch_i, batch in enumerate(loader):

            if not isinstance(batch, (list, tuple)):
                raise TypeError(
                    f"Unexpected dataloader batch type: {type(batch)}"
                )

            if len(batch) < 3:
                raise RuntimeError(
                    f"Unexpected dataloader batch length: {len(batch)}"
                )

            imgs = batch[0]
            targets = batch[1]
            paths = batch[2]

            imgs = (
                imgs.to(
                    device,
                    non_blocking=True,
                )
                .float()
                / 255.0
            )

            targets = targets.to(device)

            out = model(imgs)

            feats = extract_raw_o2o(out)

            if not isinstance(feats, (list, tuple)):
                raise TypeError(
                    f"raw O2O feats must be list/tuple, got {type(feats)}"
                )

            if len(feats) != criterion.nl:
                raise RuntimeError(
                    f"Expected {criterion.nl} O2O feature levels, got {len(feats)}"
                )

            if not printed_contract:
                print("=== Runtime contract ===")
                print("model output type:", type(out))
                print(
                    "O2O levels:",
                    [
                        tuple(x.shape)
                        for x in feats
                    ],
                )
                print(
                    "criterion:",
                    {
                        "nc": criterion.nc,
                        "nl": criterion.nl,
                        "no": criterion.no,
                        "reg_max": criterion.reg_max,
                        "o2o_topk":
                            criterion.assigner2.topk,
                        "alpha":
                            criterion.assigner2.alpha,
                        "beta":
                            criterion.assigner2.beta,
                    },
                )
                printed_contract = True

            bs = feats[0].shape[0]

            merged = torch.cat(
                [
                    xi.view(
                        bs,
                        criterion.no,
                        -1,
                    )
                    for xi in feats
                ],
                dim=2,
            )

            pred_distri, pred_scores = merged.split(
                (
                    criterion.reg_max * 4,
                    criterion.nc,
                ),
                dim=1,
            )

            pred_scores = (
                pred_scores
                .permute(0, 2, 1)
                .contiguous()
            )

            pred_distri = (
                pred_distri
                .permute(0, 2, 1)
                .contiguous()
            )

            anchor_points, stride_tensor = make_anchors(
                feats,
                criterion.stride,
                0.5,
            )

            pred_bboxes_grid = criterion.bbox_decode(
                anchor_points,
                pred_distri,
            )

            pred_bboxes_px = (
                pred_bboxes_grid
                * stride_tensor
            )

            anchor_points_px = (
                anchor_points
                * stride_tensor
            )

            # Match loss preprocessing exactly.
            imgsz_tensor = (
                torch.tensor(
                    feats[0].shape[2:],
                    device=device,
                    dtype=pred_scores.dtype,
                )
                * criterion.stride[0]
            )

            processed = criterion.preprocess(
                targets,
                bs,
                scale_tensor=imgsz_tensor[
                    [1, 0, 1, 0]
                ],
            )

            gt_labels, gt_bboxes = processed.split(
                (1, 4),
                dim=2,
            )

            mask_gt = (
                gt_bboxes
                .sum(2, keepdim=True)
                .gt_(0)
            )

            pd_scores = (
                pred_scores
                .detach()
                .sigmoid()
            )

            pd_bboxes = (
                pred_bboxes_px
                .detach()
                .type(gt_bboxes.dtype)
            )

            assigner = criterion.assigner2
            assigner.bs = bs
            assigner.n_max_boxes = gt_bboxes.size(1)

            mask_pos_pre, align_metric, ciou_overlaps = (
                assigner.get_pos_mask(
                    pd_scores,
                    pd_bboxes,
                    gt_labels,
                    gt_bboxes,
                    anchor_points_px,
                    mask_gt,
                )
            )

            (
                target_gt_idx,
                fg_mask,
                mask_pos_final,
            ) = select_highest_overlaps(
                mask_pos_pre.clone(),
                ciou_overlaps,
                gt_bboxes.size(1),
            )

            inside_mask = select_candidates_in_gts(
                anchor_points_px,
                gt_bboxes,
            ).bool()

            # Anchor -> level map.
            level_ids = []
            for level, feat in enumerate(feats):
                n = int(
                    feat.shape[-2]
                    * feat.shape[-1]
                )
                level_ids.append(
                    torch.full(
                        (n,),
                        level,
                        device=device,
                        dtype=torch.long,
                    )
                )

            level_ids = torch.cat(
                level_ids,
                dim=0,
            )

            max_scores, pred_labels = pd_scores.max(
                dim=-1
            )

            for bi in range(bs):

                if image_counter >= args.max_images:
                    break

                valid_gts = torch.where(
                    mask_gt[bi, :, 0]
                )[0]

                if len(valid_gts) == 0:
                    image_counter += 1
                    continue

                gt_boxes_i = gt_bboxes[
                    bi,
                    valid_gts,
                ]

                std_iou = standard_box_iou(
                    gt_boxes_i,
                    pd_bboxes[bi],
                )

                for local_gi, gi_tensor in enumerate(valid_gts):

                    gi = int(gi_tensor.item())
                    gt_cls = int(
                        gt_labels[
                            bi,
                            gi,
                            0,
                        ].item()
                    )

                    inside_idx = torch.where(
                        inside_mask[
                            bi,
                            gi,
                        ]
                    )[0]

                    align_info = top2(
                        align_metric[
                            bi,
                            gi,
                        ],
                        inside_idx,
                    )

                    gtclass_info = top2(
                        pd_scores[
                            bi,
                            :,
                            gt_cls,
                        ],
                        inside_idx,
                    )

                    std_iou_g = std_iou[
                        local_gi
                    ]

                    all_idx = torch.arange(
                        std_iou_g.numel(),
                        device=device,
                    )

                    best_iou_info = top2(
                        std_iou_g,
                        all_idx,
                    )

                    pre_idx = torch.where(
                        mask_pos_pre[
                            bi,
                            gi,
                        ] > 0
                    )[0]

                    final_idx = torch.where(
                        mask_pos_final[
                            bi,
                            gi,
                        ] > 0
                    )[0]

                    selected_before = int(
                        len(pre_idx) > 0
                    )

                    selected_after = int(
                        len(final_idx) > 0
                    )

                    pre_anchor = (
                        int(pre_idx[0].item())
                        if selected_before
                        else -1
                    )

                    selected_anchor = (
                        int(final_idx[0].item())
                        if selected_after
                        else -1
                    )

                    lost_by_conflict = int(
                        selected_before
                        and not selected_after
                    )

                    pre_multi_gt = 0

                    if selected_before:
                        pre_multi_gt = int(
                            mask_pos_pre[
                                bi,
                                :,
                                pre_anchor,
                            ].sum().item()
                            > 1
                        )

                    if selected_after:
                        selected_level = int(
                            level_ids[
                                selected_anchor
                            ].item()
                        )

                        selected_std_iou = float(
                            std_iou_g[
                                selected_anchor
                            ].item()
                        )

                        selected_ciou = float(
                            ciou_overlaps[
                                bi,
                                gi,
                                selected_anchor,
                            ].item()
                        )

                        selected_gtclass = float(
                            pd_scores[
                                bi,
                                selected_anchor,
                                gt_cls,
                            ].item()
                        )

                        selected_maxscore = float(
                            max_scores[
                                bi,
                                selected_anchor,
                            ].item()
                        )

                        selected_pred_class = int(
                            pred_labels[
                                bi,
                                selected_anchor,
                            ].item()
                        )

                        selected_pred_class_correct = int(
                            selected_pred_class
                            == gt_cls
                        )
                    else:
                        selected_level = -1
                        selected_std_iou = 0.0
                        selected_ciou = 0.0
                        selected_gtclass = 0.0
                        selected_maxscore = 0.0
                        selected_pred_class = -1
                        selected_pred_class_correct = 0

                    # High-IoU inference competitors.
                    iou30_idx = torch.where(
                        std_iou_g >= 0.3
                    )[0]

                    iou50_idx = torch.where(
                        std_iou_g >= 0.5
                    )[0]

                    iou50_maxscore_info = top2(
                        max_scores[bi],
                        iou50_idx,
                    )

                    iou50_gtclass_info = top2(
                        pd_scores[
                            bi,
                            :,
                            gt_cls,
                        ],
                        iou50_idx,
                    )

                    # Scale distribution among IoU>=0.5 candidates.
                    level_counts = []

                    for level in range(3):
                        level_counts.append(
                            int(
                                (
                                    level_ids[
                                        iou50_idx
                                    ]
                                    == level
                                ).sum().item()
                            )
                        )

                    iou50_scale_count = int(
                        sum(
                            int(x > 0)
                            for x in level_counts
                        )
                    )

                    iou50_has_cross_scale = int(
                        iou50_scale_count >= 2
                    )

                    same_scale_comp = 0
                    cross_scale_comp = 0

                    best_comp_gtclass = 0.0
                    best_comp_maxscore = 0.0

                    if selected_after and len(iou50_idx):

                        comp_idx = iou50_idx[
                            iou50_idx
                            != selected_anchor
                        ]

                        if len(comp_idx):
                            comp_levels = level_ids[
                                comp_idx
                            ]

                            same_scale_comp = int(
                                (
                                    comp_levels
                                    == selected_level
                                ).sum().item()
                            )

                            cross_scale_comp = int(
                                (
                                    comp_levels
                                    != selected_level
                                ).sum().item()
                            )

                            best_comp_gtclass = float(
                                pd_scores[
                                    bi,
                                    comp_idx,
                                    gt_cls,
                                ].max().item()
                            )

                            best_comp_maxscore = float(
                                max_scores[
                                    bi,
                                    comp_idx,
                                ].max().item()
                            )

                    selected_minus_best_comp_gtclass = (
                        selected_gtclass
                        - best_comp_gtclass
                    )

                    selected_minus_best_comp_maxscore = (
                        selected_maxscore
                        - best_comp_maxscore
                    )

                    selected_gtclass_wins = int(
                        selected_after
                        and selected_std_iou >= 0.5
                        and selected_minus_best_comp_gtclass >= 0
                    )

                    selected_maxscore_wins = int(
                        selected_after
                        and selected_std_iou >= 0.5
                        and selected_minus_best_comp_maxscore >= 0
                    )

                    selected_matches_inf_maxscore = int(
                        selected_after
                        and selected_anchor
                        == iou50_maxscore_info["top1_idx"]
                    )

                    selected_matches_inf_gtclass = int(
                        selected_after
                        and selected_anchor
                        == iou50_gtclass_info["top1_idx"]
                    )

                    selected_equals_best_iou = int(
                        selected_after
                        and selected_anchor
                        == best_iou_info["top1_idx"]
                    )

                    selected_equals_best_gtclass_inside = int(
                        selected_after
                        and selected_anchor
                        == gtclass_info["top1_idx"]
                    )

                    top2_align_same_level = int(
                        align_info["top1_idx"] >= 0
                        and align_info["top2_idx"] >= 0
                        and int(
                            level_ids[
                                align_info["top1_idx"]
                            ].item()
                        )
                        == int(
                            level_ids[
                                align_info["top2_idx"]
                            ].item()
                        )
                    )

                    rows.append({
                        "tag": args.tag,
                        "image": Path(paths[bi]).name,
                        "image_index": image_counter,
                        "batch_index": batch_i,
                        "gt_local_index": local_gi,
                        "gt_padded_index": gi,
                        "gt_class": gt_cls,

                        "inside_anchor_count":
                            int(len(inside_idx)),

                        "align_top1": align_info["top1"],
                        "align_top2": align_info["top2"],
                        "align_gap": align_info["gap"],
                        "align_rel_gap": align_info["rel_gap"],
                        "align_top1_anchor":
                            align_info["top1_idx"],
                        "align_top2_anchor":
                            align_info["top2_idx"],

                        "align_top1_level":
                            int(
                                level_ids[
                                    align_info["top1_idx"]
                                ].item()
                            )
                            if align_info["top1_idx"] >= 0
                            else -1,

                        "align_top2_level":
                            int(
                                level_ids[
                                    align_info["top2_idx"]
                                ].item()
                            )
                            if align_info["top2_idx"] >= 0
                            else -1,

                        "align_top2_same_level":
                            top2_align_same_level,

                        "gtclass_inside_top1":
                            gtclass_info["top1"],

                        "gtclass_inside_top2":
                            gtclass_info["top2"],

                        "gtclass_inside_gap":
                            gtclass_info["gap"],

                        "gtclass_inside_rel_gap":
                            gtclass_info["rel_gap"],

                        "selected_before_conflict":
                            selected_before,

                        "selected_after_conflict":
                            selected_after,

                        "lost_by_conflict":
                            lost_by_conflict,

                        "preselected_anchor_multi_gt":
                            pre_multi_gt,

                        "preselected_anchor":
                            pre_anchor,

                        "selected_anchor":
                            selected_anchor,

                        "selected_level":
                            selected_level,

                        "selected_stride":
                            float(
                                stride_tensor[
                                    selected_anchor
                                ].item()
                            )
                            if selected_after
                            else 0.0,

                        "selected_std_iou":
                            selected_std_iou,

                        "selected_ciou":
                            selected_ciou,

                        "selected_gtclass_score":
                            selected_gtclass,

                        "selected_maxscore":
                            selected_maxscore,

                        "selected_pred_class":
                            selected_pred_class,

                        "selected_pred_class_correct":
                            selected_pred_class_correct,

                        "best_std_iou":
                            best_iou_info["top1"],

                        "best_std_iou_anchor":
                            best_iou_info["top1_idx"],

                        "selected_equals_best_std_iou":
                            selected_equals_best_iou,

                        "selected_equals_best_gtclass_inside":
                            selected_equals_best_gtclass_inside,

                        "iou30_candidate_count":
                            int(len(iou30_idx)),

                        "iou50_candidate_count":
                            int(len(iou50_idx)),

                        "iou50_scale_count":
                            iou50_scale_count,

                        "iou50_has_cross_scale_candidates":
                            iou50_has_cross_scale,

                        "iou50_level0_count":
                            level_counts[0],

                        "iou50_level1_count":
                            level_counts[1],

                        "iou50_level2_count":
                            level_counts[2],

                        "iou50_maxscore_top1":
                            iou50_maxscore_info["top1"],

                        "iou50_maxscore_top2":
                            iou50_maxscore_info["top2"],

                        "iou50_maxscore_gap":
                            iou50_maxscore_info["gap"],

                        "iou50_maxscore_rel_gap":
                            iou50_maxscore_info["rel_gap"],

                        "iou50_gtclass_top1":
                            iou50_gtclass_info["top1"],

                        "iou50_gtclass_top2":
                            iou50_gtclass_info["top2"],

                        "iou50_gtclass_gap":
                            iou50_gtclass_info["gap"],

                        "iou50_gtclass_rel_gap":
                            iou50_gtclass_info["rel_gap"],

                        "selected_matches_inference_maxscore_iou50":
                            selected_matches_inf_maxscore,

                        "selected_matches_inference_gtclass_iou50":
                            selected_matches_inf_gtclass,

                        "same_scale_iou50_competitors":
                            same_scale_comp,

                        "cross_scale_iou50_competitors":
                            cross_scale_comp,

                        "best_competitor_gtclass_score":
                            best_comp_gtclass,

                        "best_competitor_maxscore":
                            best_comp_maxscore,

                        "selected_minus_best_comp_gtclass":
                            selected_minus_best_comp_gtclass,

                        "selected_minus_best_comp_maxscore":
                            selected_minus_best_comp_maxscore,

                        "selected_gtclass_outscores_iou50_competitors":
                            selected_gtclass_wins,

                        "selected_maxscore_outscores_iou50_competitors":
                            selected_maxscore_wins,
                    })

                image_counter += 1

            if image_counter >= args.max_images:
                break

    detail = pd.DataFrame(rows)

    if len(detail) == 0:
        raise RuntimeError(
            "No GT rows were collected."
        )

    detail_path = (
        out_dir /
        f"{args.tag}_assignment_uniqueness_detail.csv"
    )

    detail.to_csv(
        detail_path,
        index=False,
    )

    summary = summarize_rows(detail)

    summary_path = (
        out_dir /
        f"{args.tag}_assignment_uniqueness_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print("\nsaved:", detail_path)
    print("saved:", summary_path)


if __name__ == "__main__":
    main()
