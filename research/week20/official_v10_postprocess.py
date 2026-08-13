
from typing import List
import torch

from utils.e2e_postprocess import _as_bnc, select_one2one, xywh2xyxy


def official_v10_two_stage_postprocess(
    preds,
    conf_thres: float = 0.001,
    max_det: int = 300,
) -> List[torch.Tensor]:
    # YOLOv10 v10postprocess ranking adapted to this fork's
    # decoded BxNx(4+nc) one-to-one prediction format.
    #
    # Official ranking:
    # 1) top-k anchors by per-anchor max class score
    # 2) gather all class scores for selected anchors
    # 3) flatten anchor x class
    # 4) second top-k
    # 5) recover box index and class
    #
    # conf_thres is applied only after the official ranking
    # for evaluation convenience.

    pred = _as_bnc(select_one2one(preds))
    outputs = []

    for x in pred:
        if x.ndim != 2 or x.shape[1] <= 4:
            outputs.append(x.new_zeros((0, 6)))
            continue

        boxes = x[:, :4]
        scores = x[:, 4:]
        nc = scores.shape[1]

        if boxes.shape[0] == 0 or nc == 0:
            outputs.append(x.new_zeros((0, 6)))
            continue

        max_scores = scores.amax(dim=-1)
        k1 = min(int(max_det), int(max_scores.numel()))

        _, anchor_idx = torch.topk(
            max_scores,
            k=k1,
            largest=True,
            sorted=True,
        )

        selected_boxes = boxes[anchor_idx]
        selected_scores = scores[anchor_idx]

        flat = selected_scores.flatten()
        k2 = min(int(max_det), int(flat.numel()))

        top_scores, flat_idx = torch.topk(
            flat,
            k=k2,
            largest=True,
            sorted=True,
        )

        labels = flat_idx % nc
        selected_anchor_idx = torch.div(
            flat_idx,
            nc,
            rounding_mode="floor",
        )

        out_boxes = selected_boxes[selected_anchor_idx]

        keep = top_scores > float(conf_thres)

        out_boxes = out_boxes[keep]
        top_scores = top_scores[keep]
        labels = labels[keep]

        out_boxes = xywh2xyxy(out_boxes)

        outputs.append(
            torch.cat(
                (
                    out_boxes,
                    top_scores[:, None],
                    labels[:, None].to(out_boxes.dtype),
                ),
                dim=1,
            )
        )

    return outputs
