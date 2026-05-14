from __future__ import annotations

from typing import List
import time
import torch


def _as_bnc(pred: torch.Tensor) -> torch.Tensor:
    """Normalize decoded prediction shape to [B, N, 4 + nc]."""
    if pred.ndim != 3:
        raise ValueError(f"prediction must be 3D, got shape={tuple(pred.shape)}")
    if pred.shape[1] < pred.shape[2]:
        pred = pred.transpose(1, 2).contiguous()
    if pred.shape[-1] < 5:
        raise ValueError(f"last dimension must be >= 5, got shape={tuple(pred.shape)}")
    return pred


def select_one2one(preds):
    """Select one-to-one decoded tensor from possibly nested model outputs.

    Accepted structures include:
    - Tensor
    - (Tensor, aux)
    - ([one2many_decoded, one2one_decoded], aux)
    - nested DetectMultiBackend outputs wrapping the above

    Raw dict outputs are not decoded predictions, so they are ignored during search.
    """

    def is_decoded_tensor(x):
        if not torch.is_tensor(x) or x.ndim != 3:
            return False

        # Decoded YOLO output is usually [B, 4 + nc, N] or [B, N, 4 + nc].
        return x.shape[1] >= 5 or x.shape[2] >= 5

    def choose(obj):
        if is_decoded_tensor(obj):
            return obj

        if isinstance(obj, dict):
            # Raw feature dict: {"one2many": [P3, P4, P5], "one2one": [...]}
            # This is not decoded xywh/class prediction, so do not select it.
            return None

        if isinstance(obj, (list, tuple)):
            decoded_direct = [x for x in obj if is_decoded_tensor(x)]

            # V10DualDDetect eval output: [one2many_decoded, one2one_decoded]
            if len(decoded_direct) >= 2:
                return decoded_direct[1]

            if len(decoded_direct) == 1:
                return decoded_direct[0]

            # Search nested structures, ignoring raw dict branches.
            for child in obj:
                result = choose(child)
                if result is not None:
                    return result

        return None

    selected = choose(preds)

    if selected is None:
        raise TypeError(
            "Could not find decoded one2one tensor in model output. "
            "The model may still be returning raw training outputs only."
        )

    return selected

def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def v10_no_nms_postprocess(preds, conf_thres: float = 0.001, max_det: int = 300) -> List[torch.Tensor]:
    """Top-k no-NMS postprocess for one-to-one predictions.

    Output: list of tensors with shape [num_det, 6] = [x1, y1, x2, y2, conf, cls].
    """
    pred = _as_bnc(select_one2one(preds))
    outputs: List[torch.Tensor] = []

    for x in pred:
        boxes = x[:, :4]
        cls_scores = x[:, 4:]
        scores, labels = cls_scores.max(dim=1)

        keep = scores > conf_thres
        if keep.sum() == 0:
            outputs.append(x.new_zeros((0, 6)))
            continue

        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep].to(boxes.dtype)

        k = min(max_det, scores.numel())
        if scores.numel() > k:
            top_scores, top_idx = torch.topk(scores, k=k, largest=True, sorted=True)
            boxes = boxes[top_idx]
            labels = labels[top_idx]
            scores = top_scores
        else:
            order = scores.argsort(descending=True)
            boxes = boxes[order]
            labels = labels[order]
            scores = scores[order]

        boxes = xywh2xyxy(boxes)
        outputs.append(torch.cat((boxes, scores[:, None], labels[:, None]), dim=1))

    return outputs


def class_aware_nms_postprocess(preds, conf_thres: float = 0.001, iou_thres: float = 0.7, max_det: int = 300) -> List[torch.Tensor]:
    """Reference class-aware NMS path for latency comparison."""
    from torchvision.ops import nms

    pred = _as_bnc(select_one2one(preds))
    outputs: List[torch.Tensor] = []
    max_wh = 7680.0

    for x in pred:
        boxes = x[:, :4]
        cls_scores = x[:, 4:]
        scores, labels = cls_scores.max(dim=1)

        keep = scores > conf_thres
        if keep.sum() == 0:
            outputs.append(x.new_zeros((0, 6)))
            continue

        boxes = xywh2xyxy(boxes[keep])
        scores = scores[keep]
        labels = labels[keep]

        offsets = labels.to(boxes.dtype).view(-1, 1) * max_wh
        keep_idx = nms(boxes + offsets, scores, iou_thres)[:max_det]

        outputs.append(
            torch.cat(
                (boxes[keep_idx], scores[keep_idx, None], labels[keep_idx, None].to(boxes.dtype)),
                dim=1,
            )
        )

    return outputs


def profile_function(fn, preds, repeat: int = 100, warmup: int = 10, cuda_sync: bool = True) -> float:
    device = select_one2one(preds).device

    for _ in range(warmup):
        _ = fn(preds)

    if cuda_sync and device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeat):
        _ = fn(preds)

    if cuda_sync and device.type == "cuda":
        torch.cuda.synchronize()

    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / repeat
