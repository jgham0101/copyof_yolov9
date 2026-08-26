from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.yolo import V10DualDormantDetect, V10DualActiveForwardDetect
from utils.general import non_max_suppression


def tload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def unwrap(ckpt):
    if isinstance(ckpt, dict):
        return ckpt.get("ema") or ckpt.get("model")
    return ckpt


def compare(name, a, b):
    if a.shape != b.shape:
        return {
            "name": name,
            "shape_a": str(list(a.shape)),
            "shape_b": str(list(b.shape)),
            "shape_equal": False,
            "max_abs_diff": float("inf"),
            "mean_abs_diff": float("inf"),
            "allclose": False,
        }

    af = a.float()
    bf = b.float()
    diff = (af - bf).abs()

    return {
        "name": name,
        "shape_a": str(list(a.shape)),
        "shape_b": str(list(b.shape)),
        "shape_equal": True,
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "allclose": bool(
            torch.allclose(
                af,
                bf,
                rtol=1e-5,
                atol=1e-6,
            )
        ),
    }


def feature_hooks(model, store):
    hs = []

    for idx in [15, 18, 21]:
        def hook(_m, _inp, out, idx=idx):
            store[idx] = out.detach().clone()

        hs.append(
            model.model[idx].register_forward_hook(hook)
        )

    return hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--active", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    ref = unwrap(tload(a.reference)).float().to(device).eval()
    active = unwrap(tload(a.active)).float().to(device).eval()

    assert isinstance(ref.model[-1], V10DualDormantDetect)
    assert isinstance(active.model[-1], V10DualActiveForwardDetect)

    # Audit mode returns O2O tensors in addition to the unchanged O2M output.
    os.environ["YOLO_WEEK27_AUDIT_RETURN_O2O"] = "1"

    # Count every per-scale O2O bbox/cls head execution.
    calls = {
        "one2one_cv2_per_scale": [0, 0, 0],
        "one2one_cv3_per_scale": [0, 0, 0],
    }

    call_handles = []

    for i, mod in enumerate(active.model[-1].one2one_cv2):
        def hook(_m, _inp, _out, i=i):
            calls["one2one_cv2_per_scale"][i] += 1
        call_handles.append(mod.register_forward_hook(hook))

    for i, mod in enumerate(active.model[-1].one2one_cv3):
        def hook(_m, _inp, _out, i=i):
            calls["one2one_cv3_per_scale"][i] += 1
        call_handles.append(mod.register_forward_hook(hook))

    ref_feat = {}
    active_feat = {}

    ref_handles = feature_hooks(ref, ref_feat)
    active_handles = feature_hooks(active, active_feat)

    torch.manual_seed(2027)
    torch.cuda.manual_seed_all(2027)

    x = torch.rand(
        1, 3, 640, 640,
        device=device,
    )

    with torch.no_grad():
        ref_out = ref(x)
        active_bundle = active(x.clone())

    for h in call_handles + ref_handles + active_handles:
        h.remove()

    del os.environ["YOLO_WEEK27_AUDIT_RETURN_O2O"]

    # Reference DDetect output.
    ref_dec = ref_out[0]
    ref_raw = ref_out[1]

    # Active audit bundle.
    active_o2m = active_bundle["o2m"]
    active_o2m_dec = active_o2m[0]
    active_o2m_raw = active_o2m[1]

    active_o2o_raw = active_bundle["o2o_raw"]
    active_o2o_dec = active_bundle["o2o_decoded"]

    # --------------------------------------------------------
    # G2 forward call
    # --------------------------------------------------------
    all_call_counts = (
        calls["one2one_cv2_per_scale"]
        + calls["one2one_cv3_per_scale"]
    )

    call_report = {
        **calls,
        "total_o2o_child_calls": int(sum(all_call_counts)),
        "expected_total_calls": 6,
        "all_six_children_called_once": bool(
            all(v == 1 for v in all_call_counts)
        ),
    }

    (out_dir / "week27_o2o_forward_call_report.json").write_text(
        json.dumps(call_report, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # G4 feature equivalence: Week26 vs Week27
    # --------------------------------------------------------
    feature_df = pd.DataFrame([
        compare(
            f"layer_{idx}",
            ref_feat[idx],
            active_feat[idx],
        )
        for idx in [15, 18, 21]
    ])

    feature_df.to_csv(
        out_dir / "week27_feature_equivalence.csv",
        index=False,
    )

    # --------------------------------------------------------
    # G5 Week26 O2M vs Week27 O2M raw
    # --------------------------------------------------------
    o2m_raw_df = pd.DataFrame([
        compare(
            f"P{i+3}_week26_vs_week27_o2m",
            ra,
            aa,
        )
        for i, (ra, aa) in enumerate(
            zip(ref_raw, active_o2m_raw)
        )
    ])

    o2m_raw_df.to_csv(
        out_dir / "week27_o2m_reference_equivalence.csv",
        index=False,
    )

    o2m_decode = compare(
        "week26_vs_week27_o2m_decoded",
        ref_dec,
        active_o2m_dec,
    )

    (out_dir / "week27_o2m_decode_equivalence.json").write_text(
        json.dumps(o2m_decode, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # G6 Week27 O2M vs O2O
    # --------------------------------------------------------
    branch_raw_df = pd.DataFrame([
        compare(
            f"P{i+3}_o2m_vs_o2o",
            o2m,
            o2o,
        )
        for i, (o2m, o2o) in enumerate(
            zip(active_o2m_raw, active_o2o_raw)
        )
    ])

    branch_raw_df.to_csv(
        out_dir / "week27_o2m_vs_o2o_raw.csv",
        index=False,
    )

    branch_decode = compare(
        "week27_o2m_vs_o2o_decoded",
        active_o2m_dec,
        active_o2o_dec,
    )

    (out_dir / "week27_o2m_vs_o2o_decode.json").write_text(
        json.dumps(branch_decode, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # NMS: reference vs Week27 O2M, and Week27 O2M vs O2O
    # --------------------------------------------------------
    ref_nms = non_max_suppression(
        ref_dec.clone(),
        conf_thres=0.001,
        iou_thres=0.7,
        multi_label=True,
        max_det=300,
    )

    active_o2m_nms = non_max_suppression(
        active_o2m_dec.clone(),
        conf_thres=0.001,
        iou_thres=0.7,
        multi_label=True,
        max_det=300,
    )

    active_o2o_nms = non_max_suppression(
        active_o2o_dec.clone(),
        conf_thres=0.001,
        iou_thres=0.7,
        multi_label=True,
        max_det=300,
    )

    nms_rows = []

    for i, (rn, on, oo) in enumerate(
        zip(ref_nms, active_o2m_nms, active_o2o_nms)
    ):
        ref_o2m = compare(
            f"batch_{i}_week26_vs_week27_o2m_nms",
            rn,
            on,
        )
        o2m_o2o = compare(
            f"batch_{i}_week27_o2m_vs_o2o_nms",
            on,
            oo,
        )

        nms_rows.append({
            "batch_index": i,
            "week26_count": int(rn.shape[0]),
            "week27_o2m_count": int(on.shape[0]),
            "week27_o2o_count": int(oo.shape[0]),
            "reference_vs_o2m_max_abs_diff":
                ref_o2m["max_abs_diff"],
            "reference_vs_o2m_allclose":
                ref_o2m["allclose"],
            "o2m_vs_o2o_max_abs_diff":
                o2m_o2o["max_abs_diff"],
            "o2m_vs_o2o_allclose":
                o2m_o2o["allclose"],
        })

    nms_df = pd.DataFrame(nms_rows)

    nms_df.to_csv(
        out_dir / "week27_nms_equivalence.csv",
        index=False,
    )

    gate = {
        "o2o_forward_pass":
            call_report["all_six_children_called_once"],
        "feature_pass":
            bool(feature_df["allclose"].all()),
        "o2m_raw_reference_pass":
            bool(o2m_raw_df["allclose"].all()),
        "o2m_decode_reference_pass":
            bool(o2m_decode["allclose"]),
        "o2m_vs_o2o_raw_pass":
            bool(branch_raw_df["allclose"].all()),
        "o2m_vs_o2o_decode_pass":
            bool(branch_decode["allclose"]),
        "reference_vs_o2m_nms_pass":
            bool(nms_df["reference_vs_o2m_allclose"].all()),
        "o2m_vs_o2o_nms_pass":
            bool(nms_df["o2m_vs_o2o_allclose"].all()),
        "feature_max_abs_diff":
            float(feature_df["max_abs_diff"].max()),
        "o2m_reference_raw_max_abs_diff":
            float(o2m_raw_df["max_abs_diff"].max()),
        "o2m_reference_decode_max_abs_diff":
            float(o2m_decode["max_abs_diff"]),
        "branch_raw_max_abs_diff":
            float(branch_raw_df["max_abs_diff"].max()),
        "branch_decode_max_abs_diff":
            float(branch_decode["max_abs_diff"]),
    }

    (out_dir / "week27_tensor_gate.json").write_text(
        json.dumps(gate, indent=2),
        encoding="utf-8",
    )

    print("\\n=== O2O FORWARD CALLS ===")
    print(json.dumps(call_report, indent=2))

    print("\\n=== FEATURE ===")
    print(feature_df.to_string(index=False))

    print("\\n=== WEEK26 vs WEEK27 O2M RAW ===")
    print(o2m_raw_df.to_string(index=False))

    print("\\n=== WEEK27 O2M vs O2O RAW ===")
    print(branch_raw_df.to_string(index=False))

    print("\\n=== O2M DECODE REFERENCE ===")
    print(json.dumps(o2m_decode, indent=2))

    print("\\n=== O2M vs O2O DECODE ===")
    print(json.dumps(branch_decode, indent=2))

    print("\\n=== NMS ===")
    print(nms_df.to_string(index=False))

    print("\\n=== GATE ===")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
