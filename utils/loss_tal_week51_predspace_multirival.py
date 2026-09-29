
import inspect
import os

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

        self.week51_iou = float(os.getenv("YOLO_WEEK51_PREDSPACE_IOU", "0.70"))
        self.week51_topk = int(os.getenv("YOLO_WEEK51_PREDSPACE_TOPK", "300"))
        self.week51_rivals = int(os.getenv("YOLO_WEEK51_PREDSPACE_RIVALS", "3"))
        self.week51_lambda = float(os.getenv("YOLO_WEEK51_PREDSPACE_LAMBDA", "1.0"))

        if not (0.0 < self.week51_iou <= 1.0):
            raise ValueError(self.week51_iou)
        if self.week51_topk < 2:
            raise ValueError(self.week51_topk)
        if self.week51_rivals < 1:
            raise ValueError(self.week51_rivals)
        if self.week51_lambda < 0.0:
            raise ValueError(self.week51_lambda)

        self._week51_bce = _CaptureBCE(self.o2o.BCEcls)
        self.o2o.BCEcls = self._week51_bce

        self._week51_assigner = self.o2o.assigner
        self._week51_assigner_forward = self._week51_assigner.forward
        self._week51_assigner_sig = inspect.signature(self._week51_assigner_forward)
        self._week51_assigner_last = None

        def _wrapped_assigner_forward(*args, **kwargs):
            bound = self._week51_assigner_sig.bind_partial(*args, **kwargs)
            out = self._week51_assigner_forward(*args, **kwargs)
            self._week51_assigner_last = {
                "inputs": dict(bound.arguments),
                "output": out,
            }
            return out

        self._week51_assigner.forward = _wrapped_assigner_forward

        self.week51_last_aux = None
        self.week51_last_groups = 0
        self.week51_last_selected_rivals = 0
        self.week51_last_overlap_candidates = 0
        self.week51_last_protected_fg_candidates = 0
        self.week51_last_zero_sameclass_positive_groups = 0
        self.week51_last_unmatched_representative_groups = 0
        self.week51_last_mean_rivals_per_group = 0.0

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

    def _prediction_space_multirival_aux(self):
        logits = self._week51_bce.logits
        target = self._week51_bce.target
        cap = self._week51_assigner_last

        if logits is None or target is None or cap is None:
            raise RuntimeError("Week51 capture state missing.")
        if logits.shape != target.shape or logits.ndim != 3:
            raise RuntimeError((tuple(logits.shape), tuple(target.shape)))

        out = cap["output"]
        if not isinstance(out, (list, tuple)) or len(out) not in (4, 5):
            raise RuntimeError("Unexpected O2O assigner return.")

        _, target_bboxes, target_scores, fg_mask = out[:4]
        if target_scores.shape != logits.shape:
            raise RuntimeError((tuple(target_scores.shape), tuple(logits.shape)))

        fg_mask = fg_mask.bool()
        pd_bboxes = self._find_pd_bboxes(cap["inputs"], target_bboxes.shape)

        boxes_det = pd_bboxes.detach()
        prob_det = logits.detach().sigmoid()
        fg_det = fg_mask.detach()
        target_scores_det = target_scores.detach()

        group_terms = []
        groups = 0
        selected_rivals = 0
        overlap_candidates = 0
        protected_fg_candidates = 0
        zero_sameclass_positive_groups = 0
        unmatched_representative_groups = 0

        batch_size, num_anchors, _ = logits.shape
        k = min(self.week51_topk, num_anchors)

        for bi in range(batch_size):
            anchor_score, anchor_cls = prob_det[bi].max(dim=-1)
            top_score, top_idx = torch.topk(anchor_score, k=k, largest=True, sorted=True)
            top_cls = anchor_cls[top_idx]
            top_boxes = boxes_det[bi, top_idx]
            top_fg = fg_det[bi, top_idx]
            fg_target_cls = target_scores_det[bi].argmax(dim=-1)

            iou = box_iou(top_boxes.float(), top_boxes.float())
            claimed = torch.zeros(k, dtype=torch.bool, device=logits.device)

            for ri in range(k):
                if bool(claimed[ri].item()):
                    continue

                cls = int(top_cls[ri].item())

                lower = torch.arange(k, device=logits.device) > ri
                same_cls = top_cls == cls
                direct_overlap = iou[ri] >= self.week51_iou
                cand_mask = lower & (~claimed) & same_cls & direct_overlap
                cand_pos = torch.where(cand_mask)[0]

                overlap_candidates += int(cand_pos.numel())
                if cand_pos.numel() == 0:
                    continue

                member_pos = torch.cat([
                    torch.tensor([ri], device=logits.device, dtype=torch.long),
                    cand_pos
                ])
                member_anchor_idx = top_idx[member_pos]
                member_fg = fg_det[bi, member_anchor_idx]
                member_samepos = member_fg & (fg_target_cls[member_anchor_idx] == cls)

                cand_anchor_idx = top_idx[cand_pos]
                cand_fg = fg_det[bi, cand_anchor_idx]
                protected_fg_candidates += int(cand_fg.sum().item())
                eligible_pos = cand_pos[~cand_fg]

                if eligible_pos.numel() == 0:
                    continue

                chosen_pos = eligible_pos[: self.week51_rivals]
                chosen_anchor_idx = top_idx[chosen_pos]

                rival_logits = logits[bi, chosen_anchor_idx, cls]
                group_terms.append(F.softplus(rival_logits).mean())

                groups += 1
                selected_rivals += int(chosen_pos.numel())

                if not bool(member_samepos.any().item()):
                    zero_sameclass_positive_groups += 1
                if not bool(top_fg[ri].item()):
                    unmatched_representative_groups += 1

                claimed[chosen_pos] = True

        if group_terms and self.week51_lambda > 0.0:
            aux = torch.stack(group_terms).mean() * self.week51_lambda
        else:
            aux = logits.sum() * 0.0

        self.week51_last_aux = aux
        self.week51_last_groups = int(groups)
        self.week51_last_selected_rivals = int(selected_rivals)
        self.week51_last_overlap_candidates = int(overlap_candidates)
        self.week51_last_protected_fg_candidates = int(protected_fg_candidates)
        self.week51_last_zero_sameclass_positive_groups = int(zero_sameclass_positive_groups)
        self.week51_last_unmatched_representative_groups = int(unmatched_representative_groups)
        self.week51_last_mean_rivals_per_group = (
            float(selected_rivals / groups) if groups else 0.0
        )
        return aux

    def __call__(self, p, targets, img=None, epoch=0):
        total, items = super().__call__(p, targets, img=img, epoch=epoch)
        aux = self._prediction_space_multirival_aux()
        total = total + aux
        return total, items


ComputeLoss = ComputeLossWeek28V2
