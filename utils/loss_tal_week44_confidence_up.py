import json
import os
from pathlib import Path

import torch
import torch.nn as nn

from utils.loss_tal_week39_cls2 import (
    ComputeLossWeek28V2 as _Week39Loss
)


class _CaptureBCE(nn.Module):

    def __init__(self, base):

        super().__init__()

        self.base = base
        self.logits = None
        self.target = None


    def forward(
        self,
        logits,
        target,
    ):

        out = self.base(
            logits,
            target,
        )

        self.logits = logits
        self.target = target

        return out


class ComputeLossWeek28V2(_Week39Loss):

    """
    Week39 O2O cls×2
    +
    one Week44 auxiliary term only.

    Existing O2O positive target entry:

        y = target_score
        p = sigmoid(o2o_logit)

        gap = ReLU(y - p)

        aux =
            mean(gap^2)
            * batch_size
            * lambda

    Therefore:

    - only under-confident O2O positives get extra UP pressure
    - if p >= target_score, auxiliary penalty = 0
    - O2M unchanged
    - existing O2O BCE unchanged
    - negative BCE unchanged
    - box / DFL unchanged
    - assigner / target scores unchanged
    - head / decode / NMS unchanged
    """


    def __init__(
        self,
        model,
    ):

        super().__init__(
            model
        )


        self.week44_lambda = float(
            os.getenv(
                "YOLO_WEEK44_CONF_AUX_LAMBDA",
                "1.0",
            )
        )


        if (
            self.week44_lambda
            < 0
        ):

            raise ValueError(
                self.week44_lambda
            )


        # Observe the already-existing
        # O2O BCE logits and target_scores.
        #
        # The underlying BCE calculation
        # itself is not changed.

        self._cap_o2o = (
            _CaptureBCE(
                self.o2o.BCEcls
            )
        )


        self.o2o.BCEcls = (
            self._cap_o2o
        )


        self.week44_last_aux = (
            None
        )


        self.week44_last_positive_entries = (
            0
        )


        self.week44_calls = (
            0
        )


        self.telemetry_path = (
            os.getenv(
                "YOLO_WEEK44_AUX_TELEMETRY",
                "",
            )
        )


        self.telemetry_every = int(
            os.getenv(
                "YOLO_WEEK44_AUX_TELEMETRY_EVERY",
                "50",
            )
        )


    def __call__(
        self,
        p,
        targets,
        img=None,
        epoch=0,
    ):

        # ----------------------------------------------------
        # Existing Week39 dual loss
        # ----------------------------------------------------

        total, items = (
            super().__call__(
                p,
                targets,
                img=img,
                epoch=epoch,
            )
        )


        # ----------------------------------------------------
        # Captured O2O classification tensors
        # ----------------------------------------------------

        ol = (
            self._cap_o2o.logits
        )


        ot = (
            self._cap_o2o.target
        )


        if (
            ol is None
            or ot is None
        ):

            raise RuntimeError(
                "Week44 O2O BCE capture missing."
            )


        if (
            ol.shape
            != ot.shape
        ):

            raise RuntimeError(
                (
                    ol.shape,
                    ot.shape,
                )
            )


        # ----------------------------------------------------
        # O2O positive entries only
        # ----------------------------------------------------

        pos = (
            ot > 0
        )


        self.week44_last_positive_entries = int(
            pos.sum().item()
        )


        # ----------------------------------------------------
        # Target-conditioned confidence-UP auxiliary
        # ----------------------------------------------------

        if (
            bool(
                pos.any()
            )
            and self.week44_lambda > 0
        ):

            p_o2o = (
                ol.sigmoid()
            )


            gap = torch.relu(
                ot[pos]
                - p_o2o[pos]
            )


            aux = (
                gap.square().mean()
                * ol.shape[0]
                * self.week44_lambda
            )


        else:

            # Graph-connected exact zero.
            aux = (
                ol.sum()
                * 0.0
            )


        self.week44_last_aux = (
            aux
        )


        total = (
            total
            + aux
        )


        # ----------------------------------------------------
        # Telemetry
        # ----------------------------------------------------

        self.week44_calls += 1


        if (
            self.telemetry_path
            and self.telemetry_every > 0
            and (
                self.week44_calls
                % self.telemetry_every
                == 0
            )
        ):

            path = Path(
                self.telemetry_path
            )


            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )


            rec = {
                "call":
                    self.week44_calls,

                "epoch":
                    int(
                        epoch
                    ),

                "lambda":
                    self.week44_lambda,

                "positive_entries":
                    self.week44_last_positive_entries,

                "aux":
                    float(
                        aux.detach().cpu()
                    ),
            }


            with path.open(
                "a",
                encoding="utf-8",
            ) as f:

                f.write(
                    json.dumps(
                        rec
                    )
                    + "\n"
                )


        return (
            total,
            items,
        )


ComputeLoss = ComputeLossWeek28V2
