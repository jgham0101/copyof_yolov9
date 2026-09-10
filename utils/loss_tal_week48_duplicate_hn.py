
import inspect
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import box_iou
from utils.loss_tal_week39_cls2 import ComputeLossWeek28V2 as _Week39Loss


class _CaptureBCE(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.logits = None
        self.target = None

    def forward(self, logits, target):
        out = self.base(logits, target)
        self.logits = logits
        self.target = target
        return out


class ComputeLossWeek28V2(_Week39Loss):
    def __init__(self, model):
        super().__init__(model)

        self.week48_iou = float(os.getenv("YOLO_WEEK48_DUP_HN_IOU", "0.70"))
        self.week48_lambda = float(os.getenv("YOLO_WEEK48_DUP_HN_LAMBDA", "1.0"))

        if not (0.0 < self.week48_iou <= 1.0):
            raise ValueError(self.week48_iou)
        if self.week48_lambda < 0.0:
            raise ValueError(self.week48_lambda)

        # Capture graph-connected O2O classification logits/targets.
        self._week48_bce = _CaptureBCE(self.o2o.BCEcls)
        self.o2o.BCEcls = self._week48_bce

        # Capture O2O assigner inputs/output without changing assignment.
        self._week48_assigner = self.o2o.assigner
        self._week48_assigner_forward = self._week48_assigner.forward
        self._week48_assigner_sig = inspect.signature(self._week48_assigner_forward)
        self._week48_assigner_last = None

        def _wrapped_assigner_forward(*args, **kwargs):
            bound = self._week48_assigner_sig.bind_partial(*args, **kwargs)
            out = self._week48_assigner_forward(*args, **kwargs)
            self._week48_assigner_last = {
                "inputs": dict(bound.arguments),
                "output": out,
            }
            return out

        self._week48_assigner.forward = _wrapped_assigner_forward

        self.week48_last_aux = None
        self.week48_last_hard_pairs = 0
        self.week48_last_overlap_candidates = 0
        self.week48_last_positive_anchors = 0
        self.week48_last_hard_score_mean = None
        self.week48_calls = 0

        self.telemetry_path = os.getenv("YOLO_WEEK48_DUP_HN_TELEMETRY", "")
        self.telemetry_every = int(
            os.getenv("YOLO_WEEK48_DUP_HN_TELEMETRY_EVERY", "50")
        )

    @staticmethod
    def _find_pd_bboxes(inputs, expected_shape):
        if "pd_bboxes" in inputs and torch.is_tensor(inputs["pd_bboxes"]):
            x = inputs["pd_bboxes"]
            if tuple(x.shape) == tuple(expected_shape):
                return x

        candidates = []
        for k, v in inputs.items():
            if (
                torch.is_tensor(v)
                and v.ndim == 3
                and v.shape[-1] == 4
                and tuple(v.shape) == tuple(expected_shape)
            ):
                candidates.append((k, v))

        if len(candidates) != 1:
            raise RuntimeError(
                "Could not uniquely identify pd_bboxes: "
                + str([(k, tuple(v.shape)) for k, v in candidates])
            )
        return candidates[0][1]

    def _duplicate_hard_negative_aux(self):
        logits = self._week48_bce.logits
        target = self._week48_bce.target
        cap = self._week48_assigner_last

        if logits is None or target is None:
            raise RuntimeError("Week48 O2O BCE capture missing.")
        if cap is None:
            raise RuntimeError("Week48 O2O assigner capture missing.")
        if logits.shape != target.shape or logits.ndim != 3:
            raise RuntimeError((tuple(logits.shape), tuple(target.shape)))

        out = cap["output"]
        if not isinstance(out, (list, tuple)) or len(out) not in (4, 5):
            raise RuntimeError(
                "Unexpected O2O TaskAlignedAssigner return: "
                + str(type(out)) + " / " + str(len(out) if hasattr(out, "__len__") else None)
            )

        target_labels, target_bboxes, target_scores, fg_mask = out[:4]

        if target_scores.shape != logits.shape:
            raise RuntimeError(
                ("target_scores/logits shape mismatch",
                 tuple(target_scores.shape), tuple(logits.shape))
            )

        fg_mask = fg_mask.bool()
        if fg_mask.shape != logits.shape[:2]:
            raise RuntimeError((tuple(fg_mask.shape), tuple(logits.shape)))

        pd_bboxes = self._find_pd_bboxes(
            cap["inputs"],
            target_bboxes.shape,
        )

        # Selection geometry is explicitly stop-gradient.
        pd_boxes_det = pd_bboxes.detach()
        target_scores_det = target_scores.detach()
        fg_det = fg_mask.detach()

        aux_terms = []
        hard_scores = []
        hard_pairs = 0
        overlap_candidates = 0
        positive_anchors = int(fg_det.sum().item())

        batch_size = int(logits.shape[0])

        for bi in range(batch_size):
            pos_idx = torch.where(fg_det[bi])[0]
            neg_idx = torch.where(~fg_det[bi])[0]

            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue

            pos_boxes = pd_boxes_det[bi, pos_idx]
            neg_boxes = pd_boxes_det[bi, neg_idx]

            # IoU is used only for hard-negative selection.
            iou = box_iou(pos_boxes.float(), neg_boxes.float())

            pos_cls = target_scores_det[bi, pos_idx].argmax(dim=-1).long()

            for pj in range(pos_idx.numel()):
                cand_local = torch.where(iou[pj] >= self.week48_iou)[0]
                overlap_candidates += int(cand_local.numel())

                if cand_local.numel() == 0:
                    continue

                cls = int(pos_cls[pj].item())
                cand_anchor_idx = neg_idx[cand_local]

                # Select the strongest unmatched rival for THIS target class.
                cand_logits = logits[bi, cand_anchor_idx, cls]
                hard_local = int(torch.argmax(cand_logits.detach()).item())
                hard_logit = cand_logits[hard_local]

                # BCEWithLogits(target=0) == softplus(logit).
                aux_terms.append(F.softplus(hard_logit))
                hard_scores.append(hard_logit.detach().sigmoid())
                hard_pairs += 1

        if aux_terms and self.week48_lambda > 0.0:
            aux_sum = torch.stack(aux_terms).sum()
            denom = target.detach().sum().clamp_min(1.0)
            aux = (
                aux_sum / denom
                * float(batch_size)
                * self.week48_lambda
            )
        else:
            aux = logits.sum() * 0.0

        self.week48_last_aux = aux
        self.week48_last_hard_pairs = int(hard_pairs)
        self.week48_last_overlap_candidates = int(overlap_candidates)
        self.week48_last_positive_anchors = int(positive_anchors)

        if hard_scores:
            self.week48_last_hard_score_mean = float(
                torch.stack(hard_scores).mean().cpu()
            )
        else:
            self.week48_last_hard_score_mean = None

        return aux

    def __call__(self, p, targets, img=None, epoch=0):
        total, items = super().__call__(p, targets, img=img, epoch=epoch)

        aux = self._duplicate_hard_negative_aux()
        total = total + aux

        self.week48_calls += 1

        if (
            self.telemetry_path
            and self.telemetry_every > 0
            and self.week48_calls % self.telemetry_every == 0
        ):
            path = Path(self.telemetry_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            rec = {
                "call": self.week48_calls,
                "epoch": int(epoch),
                "iou": self.week48_iou,
                "lambda": self.week48_lambda,
                "positive_anchors": self.week48_last_positive_anchors,
                "overlap_candidates": self.week48_last_overlap_candidates,
                "hard_pairs": self.week48_last_hard_pairs,
                "hard_score_mean": self.week48_last_hard_score_mean,
                "aux": float(aux.detach().cpu()),
            }

            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        return total, items


ComputeLoss = ComputeLossWeek28V2
