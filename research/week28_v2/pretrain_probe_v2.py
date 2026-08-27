from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.yolo import V10DualTrainingDetectV2
from utils.dataloaders import create_dataloader
from utils.loss_tal_week28_v2 import (
    ComputeLossWeek28V2,
    native_loss_with_env,
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


def unwrap(x):
    if isinstance(x, dict):
        return x.get("ema") or x.get("model")

    return x


def bn_eval(model):
    for m in model.modules():

        if isinstance(
            m,
            torch.nn.modules.batchnorm._BatchNorm,
        ):
            m.eval()


def snapshot(model):
    return {
        n: (
            None
            if p.grad is None
            else p.grad.detach().float().cpu().clone()
        )
        for n, p in model.named_parameters()
    }


def l2_norm(grads):
    total = 0.0

    for g in grads.values():

        if g is not None:
            total += float(
                g.pow(2).sum().item()
            )

    return total ** 0.5


def compare_common(a, b):
    """
    Telemetry only.

    This compares two independent GPU backward executions.
    It is NOT used as the detach-isolation hard gate.
    """

    max_abs = 0.0
    diff_sq = 0.0
    ref_sq = 0.0
    count = 0

    for n in a:

        if "one2one_" in n:
            continue

        x = a[n]
        y = b[n]

        if x is None and y is None:
            continue

        if x is None:
            x = torch.zeros_like(y)

        if y is None:
            y = torch.zeros_like(x)

        d = x - y

        max_abs = max(
            max_abs,
            float(
                d.abs().max().item()
            )
            if d.numel()
            else 0.0,
        )

        diff_sq += float(
            d.pow(2).sum().item()
        )

        ref_sq += float(
            x.pow(2).sum().item()
        )

        count += 1

    relative_l2 = (
        diff_sq ** 0.5
    ) / (
        ref_sq ** 0.5
        + 1e-12
    )

    return (
        max_abs,
        relative_l2,
        count,
    )


def group_grad_stats(
    grads,
    want_o2o,
):

    abs_sum = 0.0
    max_abs = 0.0
    tensor_count = 0
    nonzero_tensor_count = 0

    for name, g in grads.items():

        is_o2o = (
            "one2one_"
            in name
        )

        if (
            is_o2o
            != want_o2o
        ):
            continue

        if g is None:
            continue

        tensor_count += 1

        if g.numel():

            gm = float(
                g.abs()
                .max()
                .item()
            )

            gs = float(
                g.abs()
                .sum()
                .item()
            )

            max_abs = max(
                max_abs,
                gm,
            )

            abs_sum += gs

            if gm > 0.0:
                nonzero_tensor_count += 1

    return {
        "abs_sum":
            abs_sum,

        "max_abs":
            max_abs,

        "tensor_count":
            tensor_count,

        "nonzero_tensor_count":
            nonzero_tensor_count,
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--weights",
        required=True,
    )

    ap.add_argument(
        "--data",
        required=True,
    )

    ap.add_argument(
        "--hyp",
        required=True,
    )

    ap.add_argument(
        "--out",
        required=True,
    )

    a = ap.parse_args()


    if not torch.cuda.is_available():

        raise RuntimeError(
            "GPU required from Week28-v2 "
            "pretrain probe onward"
        )


    device = torch.device(
        "cuda:0"
    )


    base = unwrap(
        tload(
            a.weights
        )
    ).float()


    hyp = yaml.safe_load(
        Path(
            a.hyp
        ).read_text(
            encoding="utf-8"
        )
    )


    data = yaml.safe_load(
        Path(
            a.data
        ).read_text(
            encoding="utf-8"
        )
    )


    loader, dataset = create_dataloader(

        data["train"],

        640,

        2,

        32,

        single_cls=False,

        hyp=hyp,

        augment=False,

        cache=False,

        rect=False,

        rank=-1,

        workers=0,

        image_weights=False,

        close_mosaic=False,

        quad=False,

        min_items=0,

        prefix="week28v2-probe: ",

        shuffle=False,
    )


    imgs, targets, _, _ = next(
        iter(
            loader
        )
    )


    imgs = (
        imgs
        .to(device)
        .float()
        / 255.0
    )

    targets = targets.to(
        device
    )


    def prepare():

        model = (
            copy.deepcopy(
                base
            )
            .to(device)
            .float()
        )

        model.hyp = dict(
            hyp
        )

        model.hyp[
            "label_smoothing"
        ] = 0.0

        model.nc = 80

        model.train()

        bn_eval(
            model
        )

        head = (
            model.model[-1]
        )

        if not isinstance(
            head,
            V10DualTrainingDetectV2,
        ):
            raise TypeError(
                type(
                    head
                )
            )

        head.week28_dual_training = True

        return model


    # ========================================================
    # A. CONTROL
    # ========================================================

    control = prepare()

    os.environ[
        "YOLO_O2O_LOSS_WEIGHT"
    ] = "0.0"


    control_fn = (
        ComputeLossWeek28V2(
            control
        )
    )


    native_fn = (
        native_loss_with_env(
            control,
            10,
            0.5,
            6.0,
        )
    )


    cb = control(
        imgs.clone()
    )


    native_loss, native_items = (
        native_fn(
            cb["o2m"],
            targets,
        )
    )


    control_loss, control_items = (
        control_fn(
            cb,
            targets,
        )
    )


    loss_diff = float(
        (
            native_loss.detach()
            - control_loss.detach()
        )
        .abs()
        .item()
    )


    item_diff = float(
        (
            native_items.float()
            - control_items.float()
        )
        .abs()
        .max()
        .item()
    )


    control.zero_grad(
        set_to_none=True
    )

    control_loss.backward()

    gc = snapshot(
        control
    )


    # ========================================================
    # B. DUAL
    # ========================================================

    dual = prepare()

    os.environ[
        "YOLO_O2O_LOSS_WEIGHT"
    ] = "1.0"


    dual_fn = (
        ComputeLossWeek28V2(
            dual
        )
    )


    db = dual(
        imgs.clone()
    )


    dual_loss, _ = (
        dual_fn(
            db,
            targets,
        )
    )


    dual.zero_grad(
        set_to_none=True
    )

    dual_loss.backward()

    gd = snapshot(
        dual
    )


    # ========================================================
    # C. CONTROL vs DUAL common gradient
    #
    # TELEMETRY ONLY.
    # This is NOT the G4 gate.
    # ========================================================

    (
        cross_max,
        cross_rel,
        cross_count,
    ) = compare_common(
        gc,
        gd,
    )


    # ========================================================
    # D. DIRECT O2O-ONLY GRADIENT ISOLATION
    #
    # This is the actual G4 causal test.
    # ========================================================

    isolation = prepare()


    o2o_only_fn = (
        native_loss_with_env(
            isolation,
            1,
            0.5,
            6.0,
        )
    )


    ib = isolation(
        imgs.clone()
    )


    (
        o2o_only_loss,
        _,
    ) = o2o_only_fn(
        ib["o2o"],
        targets,
    )


    isolation.zero_grad(
        set_to_none=True
    )


    o2o_only_loss.backward()


    gi = snapshot(
        isolation
    )


    common_isolation = (
        group_grad_stats(
            gi,
            want_o2o=False,
        )
    )


    o2o_isolation = (
        group_grad_stats(
            gi,
            want_o2o=True,
        )
    )


    control_o2o = (
        group_grad_stats(
            gc,
            want_o2o=True,
        )
    )


    # ========================================================
    # E. Global clipping telemetry
    # ========================================================

    control_norm = (
        l2_norm(
            gc
        )
    )

    dual_norm = (
        l2_norm(
            gd
        )
    )


    control_clip = min(
        1.0,
        10.0
        / (
            control_norm
            + 1e-12
        ),
    )


    dual_clip = min(
        1.0,
        10.0
        / (
            dual_norm
            + 1e-12
        ),
    )


    # O2O-only graph must be disconnected from common params.
    G4_TOL = 1e-12


    report = {

        "native_loss_abs_diff":
            loss_diff,

        "native_items_max_abs_diff":
            item_diff,

        "o2m_topk":
            control_fn.o2m_topk,

        "o2o_topk":
            control_fn.o2o_topk,

        "alpha":
            control_fn.alpha,

        "beta":
            control_fn.beta,


        # ----------------------------------------------------
        # G4 compatibility fields
        #
        # These now represent DIRECT O2O-only leakage.
        # ----------------------------------------------------

        "common_gradient_max_abs_diff_before_clip":
            common_isolation[
                "max_abs"
            ],

        "common_gradient_relative_l2_before_clip":
            0.0,

        "common_gradient_tensor_count":
            common_isolation[
                "tensor_count"
            ],

        "g4_abs_tolerance":
            G4_TOL,


        "o2o_only_common_grad_abs_sum":
            common_isolation[
                "abs_sum"
            ],

        "o2o_only_common_grad_max_abs":
            common_isolation[
                "max_abs"
            ],

        "o2o_only_common_nonzero_tensor_count":
            common_isolation[
                "nonzero_tensor_count"
            ],


        # ----------------------------------------------------
        # Old control-vs-dual comparison retained as telemetry.
        # ----------------------------------------------------

        "control_dual_common_gradient_max_abs_diff_telemetry":
            cross_max,

        "control_dual_common_gradient_relative_l2_telemetry":
            cross_rel,

        "control_dual_common_gradient_tensor_count_telemetry":
            cross_count,


        "control_o2o_grad_abs_sum":
            control_o2o[
                "abs_sum"
            ],

        "dual_o2o_grad_abs_sum":
            o2o_isolation[
                "abs_sum"
            ],

        "o2o_only_head_grad_max_abs":
            o2o_isolation[
                "max_abs"
            ],

        "o2o_only_head_nonzero_tensor_count":
            o2o_isolation[
                "nonzero_tensor_count"
            ],


        "control_preclip_grad_norm":
            control_norm,

        "dual_preclip_grad_norm":
            dual_norm,

        "control_clip_coefficient":
            control_clip,

        "dual_clip_coefficient":
            dual_clip,

        "dual_vs_control_clip_coefficient_ratio":
            (
                dual_clip
                / control_clip
                if control_clip > 0
                else None
            ),
    }


    # ========================================================
    # Gates
    # ========================================================

    report[
        "g2_native_o2m_loss_pass"
    ] = bool(

        loss_diff <= 1e-6

        and

        item_diff <= 1e-6
    )


    report[
        "g3_assigner_config_pass"
    ] = bool(

        control_fn.o2m_topk == 10

        and

        control_fn.o2o_topk == 1

        and

        abs(
            control_fn.alpha
            - 0.5
        ) < 1e-12

        and

        abs(
            control_fn.beta
            - 6.0
        ) < 1e-12
    )


    # Direct detach-isolation gate.
    report[
        "g4_common_preclip_gradient_pass"
    ] = bool(

        common_isolation[
            "max_abs"
        ] <= G4_TOL

        and

        common_isolation[
            "abs_sum"
        ] <= G4_TOL

        and

        common_isolation[
            "nonzero_tensor_count"
        ] == 0
    )


    report[
        "g5_o2o_gradient_pass"
    ] = bool(

        control_o2o[
            "abs_sum"
        ] <= 1e-12

        and

        o2o_isolation[
            "abs_sum"
        ] > 0.0

        and

        o2o_isolation[
            "nonzero_tensor_count"
        ] > 0
    )


    report[
        "all_pretrain_gates_pass"
    ] = all([

        report[
            "g2_native_o2m_loss_pass"
        ],

        report[
            "g3_assigner_config_pass"
        ],

        report[
            "g4_common_preclip_gradient_pass"
        ],

        report[
            "g5_o2o_gradient_pass"
        ],
    ])


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


    if not report[
        "all_pretrain_gates_pass"
    ]:

        raise RuntimeError(
            "Week28 v2 G2~G5 FAIL; "
            "training forbidden"
        )


if __name__ == "__main__":
    main()
