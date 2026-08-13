
import argparse
import re
from pathlib import Path

PATCH_MARK = "# WEEK22_REP_RANK_PATCH"

METHOD_TEXT = r'''
    def _week22_rep_rank_loss(
        self,
        pred_scores2,
        pred_bboxes2,
        stride_tensor,
        target_labels2,
        target_bboxes2,
        fg_mask2,
        feats2,
    ):
        zero = pred_scores2.new_tensor(0.0)

        stats = {
            "active_positive_count": 0,
            "pair_count": 0,
            "selected_iou_mean": 0.0,
            "pre_sigmoid_gap_mean": 0.0,
            "winner_rate_before": 0.0,
        }

        if (
            self.rep_rank_weight <= 0
            and not self.rep_rank_telemetry
        ):
            return zero, stats

        if not fg_mask2.any():
            return zero, stats

        level_ids = []

        for level, feat in enumerate(feats2):
            n = int(
                feat.shape[-2]
                * feat.shape[-1]
            )

            level_ids.append(
                torch.full(
                    (n,),
                    level,
                    device=pred_scores2.device,
                    dtype=torch.long,
                )
            )

        level_ids = torch.cat(
            level_ids,
            dim=0,
        )

        pred_boxes_px = (
            pred_bboxes2.detach()
            * stride_tensor
        )

        detached_scores = (
            pred_scores2.detach()
        )

        total = zero
        active = 0

        selected_iou_sum = 0.0
        gap_sum = 0.0
        winner_sum = 0.0

        bs = pred_scores2.shape[0]

        for b in range(bs):

            pos_idx = torch.where(
                fg_mask2[b]
            )[0]

            for p_tensor in pos_idx:

                p = int(
                    p_tensor.item()
                )

                if target_labels2.ndim == 3:
                    cls = int(
                        target_labels2[
                            b, p, 0
                        ].item()
                    )
                else:
                    cls = int(
                        target_labels2[
                            b, p
                        ].item()
                    )

                if (
                    cls < 0
                    or cls >= self.nc
                ):
                    continue

                gt = target_bboxes2[
                    b, p
                ]

                boxes = pred_boxes_px[
                    b
                ]

                lt = torch.maximum(
                    boxes[:, :2],
                    gt[:2],
                )

                rb = torch.minimum(
                    boxes[:, 2:],
                    gt[2:],
                )

                wh = (
                    rb - lt
                ).clamp(min=0)

                inter = (
                    wh[:, 0]
                    * wh[:, 1]
                )

                box_area = (
                    (
                        boxes[:, 2]
                        - boxes[:, 0]
                    ).clamp(min=0)
                    *
                    (
                        boxes[:, 3]
                        - boxes[:, 1]
                    ).clamp(min=0)
                )

                gt_area = (
                    (
                        gt[2] - gt[0]
                    ).clamp(min=0)
                    *
                    (
                        gt[3] - gt[1]
                    ).clamp(min=0)
                )

                iou = (
                    inter
                    /
                    (
                        box_area
                        + gt_area
                        - inter
                        + 1e-9
                    )
                )

                pos_iou = float(
                    iou[p].item()
                )

                if (
                    pos_iou
                    < self.rep_rank_iou_threshold
                ):
                    continue

                same_scale = (
                    level_ids
                    == level_ids[p]
                )

                good_box = (
                    iou
                    >= self.rep_rank_iou_threshold
                )

                # Exclude every O2O positive from the negative pool.
                unselected = (
                    ~fg_mask2[b]
                )

                competitor_mask = (
                    same_scale
                    & good_box
                    & unselected
                )

                competitor_idx = torch.where(
                    competitor_mask
                )[0]

                if len(competitor_idx) == 0:
                    continue

                detached_neg_logits = detached_scores[
                    b,
                    competitor_idx,
                    cls,
                ]

                best_pos = int(
                    torch.argmax(
                        detached_neg_logits
                    ).item()
                )

                neg_anchor = int(
                    competitor_idx[
                        best_pos
                    ].item()
                )

                pos_logit = pred_scores2[
                    b, p, cls
                ]

                neg_logit = pred_scores2[
                    b, neg_anchor, cls
                ]

                pair_loss = torch.nn.functional.softplus(
                    neg_logit
                    - pos_logit
                )

                total = (
                    total
                    + pair_loss
                )

                active += 1
                selected_iou_sum += pos_iou

                with torch.no_grad():
                    pos_prob = float(
                        pos_logit
                        .detach()
                        .sigmoid()
                        .item()
                    )

                    neg_prob = float(
                        neg_logit
                        .detach()
                        .sigmoid()
                        .item()
                    )

                    gap_sum += (
                        pos_prob
                        - neg_prob
                    )

                    winner_sum += float(
                        pos_prob >= neg_prob
                    )

        if active <= 0:
            return zero, stats

        loss = (
            total
            / float(active)
        )

        stats = {
            "active_positive_count":
                int(active),

            "pair_count":
                int(active),

            "selected_iou_mean":
                float(
                    selected_iou_sum
                    / active
                ),

            "pre_sigmoid_gap_mean":
                float(
                    gap_sum
                    / active
                ),

            "winner_rate_before":
                float(
                    winner_sum
                    / active
                ),
        }

        return loss, stats
'''

def patch_compute_loss(text):
    if PATCH_MARK in text:
        print("already patched:", PATCH_MARK)
        return text

    class_start = text.find("class ComputeLoss:")
    if class_start < 0:
        raise RuntimeError("class ComputeLoss not found")

    next_class = text.find("\nclass ", class_start + 10)
    class_end = len(text) if next_class < 0 else next_class
    segment = text[class_start:class_end]

    cfg_block = (
        '        self.rep_rank_weight = '
        'float(os.getenv("YOLO_REP_RANK_WEIGHT", "0.0"))  '
        + PATCH_MARK + '\n'
        '        self.rep_rank_iou_threshold = '
        'float(os.getenv("YOLO_REP_RANK_IOU", "0.5"))\n'
        '        self.rep_rank_telemetry = '
        'os.getenv("YOLO_REP_RANK_TELEMETRY", "")\n'
        '        self.rep_rank_telemetry_every = '
        'max(1, int(os.getenv("YOLO_REP_RANK_TELEMETRY_EVERY", "20")))\n'
    )

    bbox_pos = segment.find("        self.bbox_loss =")
    if bbox_pos < 0:
        raise RuntimeError("self.bbox_loss not found in ComputeLoss")

    abs_bbox_pos = class_start + bbox_pos
    text = text[:abs_bbox_pos] + cfg_block + text[abs_bbox_pos:]

    class_start = text.find("class ComputeLoss:")
    next_class = text.find("\nclass ", class_start + 10)
    class_end = len(text) if next_class < 0 else next_class
    segment = text[class_start:class_end]

    call_pos = segment.find("    def __call__(self, p, targets")
    if call_pos < 0:
        raise RuntimeError("ComputeLoss.__call__ not found")

    abs_call_pos = class_start + call_pos
    text = text[:abs_call_pos] + METHOD_TEXT + "\n" + text[abs_call_pos:]

    class_start = text.find("class ComputeLoss:")
    next_class = text.find("\nclass ", class_start + 10)
    class_end = len(text) if next_class < 0 else next_class
    segment = text[class_start:class_end]

    divide_anchor = "        target_bboxes /= stride_tensor\n"
    divide_pos = segment.find(divide_anchor)
    if divide_pos < 0:
        raise RuntimeError("target_bboxes /= stride_tensor not found")

    rank_compute = r'''        # WEEK22_REP_RANK_PATCH:
        loss_rep_rank = pred_scores2.new_tensor(0.0)
        rep_rank_stats = {
            "active_positive_count": 0,
            "pair_count": 0,
            "selected_iou_mean": 0.0,
            "pre_sigmoid_gap_mean": 0.0,
            "winner_rate_before": 0.0,
        }

        if (
            self.rep_rank_weight > 0
            or self.rep_rank_telemetry
        ):
            loss_rep_rank, rep_rank_stats = self._week22_rep_rank_loss(
                pred_scores2,
                pred_bboxes2,
                stride_tensor,
                target_labels2,
                target_bboxes2,
                fg_mask2,
                feats2,
            )

        self._week22_last_rep_rank_raw = loss_rep_rank.detach()
        self._week22_last_rep_rank_stats = rep_rank_stats

        if self.rep_rank_telemetry:

            if not hasattr(self, "_week22_rep_rank_step"):
                self._week22_rep_rank_step = 0

            self._week22_rep_rank_step += 1

            if (
                self._week22_rep_rank_step
                % self.rep_rank_telemetry_every
                == 0
            ):
                try:
                    import json as _week22_json

                    _row = {
                        "step": int(self._week22_rep_rank_step),
                        "weight": float(self.rep_rank_weight),
                        "iou_threshold": float(self.rep_rank_iou_threshold),
                        "rank_loss_raw": float(
                            loss_rep_rank
                            .detach()
                            .float()
                            .cpu()
                            .item()
                        ),
                        **rep_rank_stats,
                    }

                    with open(
                        self.rep_rank_telemetry,
                        "a",
                        encoding="utf-8",
                    ) as _f:
                        _f.write(
                            _week22_json.dumps(_row)
                            + "\n"
                        )

                except Exception:
                    if (
                        os.getenv(
                            "YOLO_REP_RANK_TELEMETRY_STRICT",
                            "0",
                        )
                        == "1"
                    ):
                        raise

'''

    abs_divide_pos = class_start + divide_pos
    text = text[:abs_divide_pos] + rank_compute + text[abs_divide_pos:]

    class_start = text.find("class ComputeLoss:")
    next_class = text.find("\nclass ", class_start + 10)
    class_end = len(text) if next_class < 0 else next_class
    segment = text[class_start:class_end]

    candidate_anchors = [
        (
            "        loss[1] += loss_cls_o2o\n",
            "        loss[1] += loss_cls_o2o\n"
            "        loss[1] += self.rep_rank_weight * loss_rep_rank  "
            + PATCH_MARK + "\n",
        ),
        (
            "        loss[1] += self.BCEcls(pred_scores2, target_scores2.to(dtype)).sum() / target_scores_sum2 # BCE\n",
            "        loss[1] += self.BCEcls(pred_scores2, target_scores2.to(dtype)).sum() / target_scores_sum2 # BCE\n"
            "        loss[1] += self.rep_rank_weight * loss_rep_rank  "
            + PATCH_MARK + "\n",
        ),
        (
            "        loss[1] += self.BCEcls(pred_scores2, target_scores2.to(dtype)).sum() / target_scores_sum2  # BCE\n",
            "        loss[1] += self.BCEcls(pred_scores2, target_scores2.to(dtype)).sum() / target_scores_sum2  # BCE\n"
            "        loss[1] += self.rep_rank_weight * loss_rep_rank  "
            + PATCH_MARK + "\n",
        ),
    ]

    replaced = False

    for old, new in candidate_anchors:
        if old in segment:
            segment = segment.replace(old, new, 1)
            replaced = True
            break

    if not replaced:
        raise RuntimeError(
            "Could not locate O2O classification accumulation"
        )

    text = text[:class_start] + segment + text[class_end:]
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="utils/loss_tal_dual.py")
    ap.add_argument("--backup", default="")
    args = ap.parse_args()

    path = Path(args.file)
    assert path.exists(), path

    text = path.read_text(encoding="utf-8")

    if PATCH_MARK in text:
        print("already patched:", path)
        return

    backup = (
        Path(args.backup)
        if args.backup
        else path.with_suffix(path.suffix + ".week22_backup")
    )

    backup.write_text(text, encoding="utf-8")

    patched = patch_compute_loss(text)
    path.write_text(patched, encoding="utf-8")

    print("patched:", path)
    print("backup :", backup)


if __name__ == "__main__":
    main()
