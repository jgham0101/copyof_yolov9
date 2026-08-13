
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


VARIANTS = [
    "control_continue20",
    "pairwise_continue20",
    "positive_only_continue20",
    "negative_only_continue20",
]


def best_e2e(sweeps):
    rows = []
    x = sweeps.copy()

    for col in [
        "precision",
        "recall",
        "map50",
        "map50_95",
        "postprocess_ms",
        "conf_thres",
    ]:
        if col in x.columns:
            x[col] = pd.to_numeric(
                x[col],
                errors="coerce",
            )

    for (tag, pp), sub in x.groupby(
        ["tag", "postprocess"]
    ):
        sub = sub.dropna(subset=["map50"])

        if len(sub):
            rows.append(
                sub.sort_values(
                    ["map50", "map50_95"],
                    ascending=False,
                ).iloc[0].to_dict()
            )

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    args = ap.parse_args()

    root = Path(args.drive)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    ref_e2e = pd.read_csv(
        root /
        "reference/week22_reference_e2e.csv"
    )

    new_e2e = []

    for p in sorted(
        (root / "e2e_sweeps").glob("*_e2e_sweep.csv")
    ):
        new_e2e.append(pd.read_csv(p))

    sweeps = pd.concat(
        [ref_e2e, *new_e2e],
        ignore_index=True,
    )

    assert set(VARIANTS).issubset(
        set(sweeps["tag"])
    )

    sweeps.to_csv(
        reports /
        "week23_all_e2e_sweeps.csv",
        index=False,
    )

    best = best_e2e(sweeps)

    best.to_csv(
        reports /
        "week23_best_e2e_by_variant.csv",
        index=False,
    )

    fixed = sweeps[
        np.isclose(
            pd.to_numeric(
                sweeps["conf_thres"],
                errors="coerce",
            ),
            0.01,
        )
    ].copy()

    fixed.to_csv(
        reports /
        "week23_fixed_conf001_e2e.csv",
        index=False,
    )

    assignment = []

    for tag in VARIANTS:
        p = (
            root /
            "assignment_probe" /
            f"{tag}_assignment_uniqueness_summary.csv"
        )
        assert p.exists(), p
        assignment.append(pd.read_csv(p))

    assignment = pd.concat(
        assignment,
        ignore_index=True,
    )

    assignment.to_csv(
        reports /
        "week23_assignment_uniqueness_summary.csv",
        index=False,
    )

    box_tables = [
        pd.read_csv(
            root /
            "reference/week22_reference_box_aggregate.csv"
        )
    ]

    for p in sorted(
        (root / "box_probe").glob("*_four_mode_aggregate.csv")
    ):
        box_tables.append(pd.read_csv(p))

    box = pd.concat(
        box_tables,
        ignore_index=True,
    )

    box.to_csv(
        reports /
        "week23_box_probe_aggregate.csv",
        index=False,
    )

    bx = box.copy()
    bx["conf_thres"] = pd.to_numeric(
        bx["conf_thres"],
        errors="coerce",
    )

    fixed_box = bx[
        (bx["mode"] == "one2one_current_no_nms")
        &
        np.isclose(
            bx["conf_thres"],
            0.01,
        )
    ].copy()

    if "images" in fixed_box.columns:
        total_pred = (
            fixed_box["mean_num_pred"]
            * fixed_box["images"]
        )

        fixed_box["class_match_purity"] = (
            fixed_box["total_class_match_iou50"]
            /
            total_pred.clip(lower=1e-9)
        )

    fixed_box.to_csv(
        reports /
        "week23_fixed_conf001_box_mechanism.csv",
        index=False,
    )

    common = pd.read_csv(
        reports /
        "week23_common_gt_mechanism_summary.csv"
    )

    paired = pd.read_csv(
        reports /
        "week23_paired_mechanism_deltas.csv"
    )

    rows = []
    control_tag = "control_continue20"

    for tag in [
        "pairwise_continue20",
        "positive_only_continue20",
        "negative_only_continue20",
    ]:
        row = {
            "comparison":
                f"{tag}_minus_control",
        }

        for pp in ["nms", "no-nms"]:
            c = fixed[
                (fixed["tag"] == control_tag)
                &
                (fixed["postprocess"] == pp)
            ]

            t = fixed[
                (fixed["tag"] == tag)
                &
                (fixed["postprocess"] == pp)
            ]

            if len(c) and len(t):
                c = c.iloc[0]
                t = t.iloc[0]

                for metric in [
                    "precision",
                    "recall",
                    "map50",
                    "map50_95",
                    "postprocess_ms",
                ]:
                    row[
                        f"conf001_{pp}_{metric}_delta"
                    ] = (
                        float(t[metric])
                        - float(c[metric])
                    )

        for pp in ["nms", "no-nms"]:
            c = best[
                (best["tag"] == control_tag)
                &
                (best["postprocess"] == pp)
            ]

            t = best[
                (best["tag"] == tag)
                &
                (best["postprocess"] == pp)
            ]

            if len(c) and len(t):
                c = c.iloc[0]
                t = t.iloc[0]

                row[f"best_{pp}_map50_delta"] = (
                    float(t["map50"])
                    - float(c["map50"])
                )

                row[f"best_{pp}_map50_95_delta"] = (
                    float(t["map50_95"])
                    - float(c["map50_95"])
                )

        c = fixed_box[
            fixed_box["tag"] == control_tag
        ]

        t = fixed_box[
            fixed_box["tag"] == tag
        ]

        if len(c) and len(t):
            c = c.iloc[0]
            t = t.iloc[0]

            for metric in [
                "mean_num_pred",
                "total_class_match_iou50",
                "total_duplicate_pairs_iou70_same_class",
                "mean_conf",
                "max_conf",
                "class_match_purity",
            ]:
                if metric in c.index and metric in t.index:
                    row[f"conf001_{metric}_delta"] = (
                        float(t[metric])
                        - float(c[metric])
                    )

        c = common[
            common["tag"] == control_tag
        ]

        t = common[
            common["tag"] == tag
        ]

        if len(c) and len(t):
            c = c.iloc[0]
            t = t.iloc[0]

            for metric in [
                "selected_gtclass_winner_rate",
                "selected_maxscore_winner_rate",
                "selected_minus_comp_gtclass_median",
                "selected_minus_comp_maxscore_median",
                "selected_gtclass_score_median",
                "best_competitor_gtclass_score_median",
                "selected_maxscore_median",
                "best_competitor_maxscore_median",
                "selected_std_iou_mean",
            ]:
                row[f"{metric}_delta"] = (
                    float(t[metric])
                    - float(c[metric])
                )

        rows.append(row)

    delta = pd.DataFrame(rows)

    delta.to_csv(
        reports /
        "week23_controlled_mechanism_deltas.csv",
        index=False,
    )

    tel_rows = []

    pair_ref = (
        root /
        "reference/week22_pairwise_rank_telemetry.csv"
    )

    if pair_ref.exists():
        d = pd.read_csv(pair_ref)

        if len(d):
            tel_rows.append({
                "tag": "pairwise_continue20",
                "mode": "pairwise",
                "rows": len(d),
                "rank_loss_raw_mean":
                    float(d["rank_loss_raw"].mean()),
                "active_pairs_mean":
                    float(d["active_positive_count"].mean()),
                "winner_rate_before_mean":
                    float(d["winner_rate_before"].mean()),
                "pre_sigmoid_gap_mean":
                    float(d["pre_sigmoid_gap_mean"].mean()),
            })

    for tag, mode in [
        ("positive_only_continue20", "positive_only"),
        ("negative_only_continue20", "negative_only"),
    ]:
        p = (
            root /
            "rank_telemetry" /
            f"{tag}.jsonl"
        )

        if not p.exists():
            continue

        raw_rows = []

        for line in p.read_text(
            encoding="utf-8"
        ).splitlines():
            if line.strip():
                raw_rows.append(json.loads(line))

        d = pd.DataFrame(raw_rows)

        if len(d):
            tel_rows.append({
                "tag": tag,
                "mode": mode,
                "rows": len(d),
                "rank_loss_raw_mean":
                    float(d["rank_loss_raw"].mean()),
                "active_pairs_mean":
                    float(d["active_positive_count"].mean()),
                "winner_rate_before_mean":
                    float(d["winner_rate_before"].mean()),
                "pre_sigmoid_gap_mean":
                    float(d["pre_sigmoid_gap_mean"].mean()),
            })

    telemetry = pd.DataFrame(tel_rows)

    telemetry.to_csv(
        reports /
        "week23_gradient_mode_telemetry_summary.csv",
        index=False,
    )

    decision = pd.DataFrame([
        {
            "hypothesis":
                "H1_negative_suppression_dominant",
            "expected_pattern":
                "negative-only reproduces pairwise duplicate/AP/purity gains while winner-rate changes remain small",
            "status":
                "REQUIRES_INTERPRETATION",
        },
        {
            "hypothesis":
                "H2_positive_promotion_dominant",
            "expected_pattern":
                "positive-only improves winner/gap and reproduces downstream no-NMS gains",
            "status":
                "REQUIRES_INTERPRETATION",
        },
        {
            "hypothesis":
                "H3_pairwise_synergy",
            "expected_pattern":
                "pairwise clearly outperforms both one-sided modes",
            "status":
                "REQUIRES_INTERPRETATION",
        },
        {
            "hypothesis":
                "H4_assignment_or_calibration_feedback",
            "expected_pattern":
                "neither one-sided mode explains pairwise; inspect dynamic TAL reassignment/calibration",
            "status":
                "REQUIRES_INTERPRETATION",
        },
    ])

    decision.to_csv(
        reports /
        "week23_mechanism_decision_gate.csv",
        index=False,
    )

    lines = [
        "# Week23 — Positive Promotion vs Hard-Competitor Suppression",
        "",
        "## Best E2E",
        "",
        best.to_markdown(index=False),
        "",
        "## Fixed conf=.01 E2E",
        "",
        fixed.to_markdown(index=False),
        "",
        "## Common-GT direct mechanism",
        "",
        common.to_markdown(index=False),
        "",
        "## Paired mechanism transitions",
        "",
        paired.to_markdown(index=False),
        "",
        "## Fixed conf=.01 output mechanism",
        "",
        fixed_box.to_markdown(index=False),
        "",
        "## Controlled deltas",
        "",
        delta.to_markdown(index=False),
        "",
        "## Gradient telemetry",
        "",
        telemetry.to_markdown(index=False)
        if len(telemetry)
        else "No telemetry.",
        "",
        "## Interpretation order",
        "",
        "1. Compare positive-only and negative-only on the exact same common GT subset.",
        "2. Check whether negative-only reproduces pairwise duplicate/AP/purity improvement.",
        "3. Check whether positive-only mainly improves selected-positive score/winner rate.",
        "4. Compare both against the original pairwise result before deciding Week24.",
    ]

    (
        reports /
        "week23_summary.md"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        (
            reports /
            "week23_summary.md"
        ).read_text(encoding="utf-8")
    )


if __name__ == "__main__":
    main()
