import torch

from utils.general import xywh2xyxy


def _to_b84n(decoded, nc=80):
    # Normalize decoded prediction orientation to [B, 4+nc, N].
    if decoded.ndim != 3:
        raise ValueError(
            f"Expected rank-3 prediction tensor, got {tuple(decoded.shape)}"
        )

    c = 4 + int(nc)

    if decoded.shape[1] == c:
        return decoded.contiguous()

    if decoded.shape[2] == c:
        return decoded.permute(0, 2, 1).contiguous()

    raise ValueError(
        f"Expected [B,{c},N] or [B,N,{c}], got {tuple(decoded.shape)}"
    )


def yolov10_style_no_nms(
    decoded,
    conf_thres=0.001,
    max_det=300,
    nc=80,
):
    # YOLOv10-style two-stage top-k postprocess.
    #
    # Input:
    #   decoded:
    #       [B, 4+nc, N] or [B, N, 4+nc]
    #
    # Output:
    #   list of tensors [num_det, 6]:
    #       x1, y1, x2, y2, confidence, class
    #
    # Procedure:
    #   1. Per-anchor maximum class score
    #   2. Select top-k anchors
    #   3. Flatten all class scores for selected anchors
    #   4. Global anchor-class top-k
    #   5. Confidence threshold
    #
    # NMS is intentionally never called.

    p = _to_b84n(
        decoded,
        nc=nc,
    ).permute(
        0,
        2,
        1,
    ).contiguous()

    # p: [B, N, 4+nc]

    boxes_xywh = p[..., :4]
    scores = p[..., 4:]

    batch_size = int(p.shape[0])
    num_anchors = int(p.shape[1])

    outputs = []

    for bi in range(batch_size):

        score_i = scores[bi]
        box_i = boxes_xywh[bi]

        # -------------------------------------------------------------
        # Stage 1:
        # rank anchors using each anchor's best class confidence
        # -------------------------------------------------------------

        anchor_best_score = score_i.amax(
            dim=-1
        )

        k_anchor = min(
            int(max_det),
            num_anchors,
        )

        anchor_idx = anchor_best_score.topk(
            k=k_anchor,
            dim=-1,
            largest=True,
            sorted=True,
        ).indices


        selected_scores = score_i[
            anchor_idx
        ]

        selected_boxes = box_i[
            anchor_idx
        ]


        # -------------------------------------------------------------
        # Stage 2:
        # flatten selected anchor x class score matrix and
        # globally select the best anchor-class pairs
        # -------------------------------------------------------------

        flat_scores = selected_scores.reshape(
            -1
        )

        k_global = min(
            int(max_det),
            int(flat_scores.numel()),
        )

        top_scores, flat_idx = flat_scores.topk(
            k=k_global,
            dim=-1,
            largest=True,
            sorted=True,
        )


        local_anchor_idx = torch.div(
            flat_idx,
            int(nc),
            rounding_mode="floor",
        )

        class_idx = flat_idx.remainder(
            int(nc)
        )


        # -------------------------------------------------------------
        # Convert selected xywh boxes to xyxy
        # -------------------------------------------------------------

        boxes_xyxy = xywh2xyxy(
            selected_boxes[
                local_anchor_idx
            ]
        )


        det = torch.cat(
            (
                boxes_xyxy,

                top_scores[
                    :, None
                ],

                class_idx[
                    :, None
                ].to(
                    boxes_xyxy.dtype
                ),
            ),
            dim=1,
        )


        # -------------------------------------------------------------
        # Confidence threshold AFTER the two-stage top-k.
        #
        # This preserves the intended YOLOv10-style selection order.
        # -------------------------------------------------------------

        keep = (
            det[:, 4]
            >= float(conf_thres)
        )

        det = det[
            keep
        ]


        outputs.append(
            det
        )


    return outputs
