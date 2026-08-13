
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def best_e2e(df):
    rows = []

    if len(df) == 0:
        return pd.DataFrame()

    tmp = df.copy()

    for c in [
        "map50",
        "map50_95",
        "precision",
        "recall",
        "postprocess_ms",
        "conf_thres",
    ]:
        if c in tmp.columns:
            tmp[c] = pd.to_numeric(
                tmp[c],
                errors="coerce",
            )

    for (_, _), sub in tmp.groupby(
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
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(
        (
            root /
            "week22_experiment_config.json"
        ).read_text(encoding="utf-8")
    )

    train_rows = []

    for tag, spec in config["variants"].items():
        run = (
            root /
            "runs_backup" /
            spec["run_name"]
        )

        results = run / "results.csv"

        row = {
            "tag": tag,
            "run_name": spec["run_name"],
        }

        if results.exists():
            df = pd.read_csv(results)

            row["epochs_recorded"] = len(df)

            aliases = {
                "precision": [
                    "metrics/precision",
                    "metrics/precision(B)",
                ],
                "recall": [
                    "metrics/recall",
                    "metrics/recall(B)",
                ],
                "map50": [
                    "metrics/mAP_0.5",
                    "metrics/mAP50(B)",
                ],
                "map50_95": [
                    "metrics/mAP_0.5:0.95",
                    "metrics/mAP50-95(B)",
                ],
            }

            for name, cands in aliases.items():
                col = next(
                    (c for c in cands if c in df.columns),
                    None,
                )

                if col is None:
                    continue

                s = pd.to_numeric(
                    df[col],
                    errors="coerce",
                )

                row[f"final_{name}"] = (
                    float(s.iloc[-1])
                    if len(s)
                    else None
                )

                row[f"best_{name}"] = (
                    float(s.max())
                    if s.notna().any()
                    else None
                )

        train_rows.append(row)

    train = pd.DataFrame(train_rows)

    train.to_csv(
        report_dir /
        "week22_training_summary.csv",
        index=False,
    )

    sweep_tables = []

    for p in sorted(
        (root / "e2e_sweeps").glob("*_e2e_sweep.csv")
    ):
        x = pd.read_csv(p)
        x["source_file"] = p.name
        sweep_tables.append(x)

    sweeps = (
        pd.concat(sweep_tables, ignore_index=True)
        if sweep_tables
        else pd.DataFrame()
    )

    sweeps.to_csv(
        report_dir /
        "week22_all_e2e_sweeps.csv",
        index=False,
    )

    best = best_e2e(sweeps)

    best.to_csv(
        report_dir /
        "week22_best_e2e_by_variant.csv",
        index=False,
    )

    assign_tables = []

    for p in sorted(
        (root / "assignment_probe").glob(
            "*_assignment_uniqueness_summary.csv"
        )
    ):
        assign_tables.append(pd.read_csv(p))

    assignment = (
        pd.concat(assign_tables, ignore_index=True)
        if assign_tables
        else pd.DataFrame()
    )

    assignment.to_csv(
        report_dir /
        "week22_assignment_uniqueness_summary.csv",
        index=False,
    )

    box_tables = []

    for p in sorted(
        (root / "box_probe").glob("*_four_mode_aggregate.csv")
    ):
        x = pd.read_csv(p)
        x["source_file"] = p.name
        box_tables.append(x)

    box = (
        pd.concat(box_tables, ignore_index=True)
        if box_tables
        else pd.DataFrame()
    )

    box.to_csv(
        report_dir /
        "week22_box_probe_aggregate.csv",
        index=False,
    )

    telemetry_path = (
        root /
        "rank_telemetry/rank_continue20.jsonl"
    )

    telemetry_rows = []

    if telemetry_path.exists():
        for line in telemetry_path.read_text(
            encoding="utf-8"
        ).splitlines():
            if line.strip():
                telemetry_rows.append(
                    json.loads(line)
                )

    telemetry = pd.DataFrame(telemetry_rows)

    telemetry.to_csv(
        report_dir /
        "week22_rank_telemetry.csv",
        index=False,
    )

    delta = {
        "comparison":
            "rank_continue20_minus_control_continue20",
    }

    control_tag = "control_continue20"
    rank_tag = "rank_continue20"

    for pp in ["nms", "no-nms"]:

        c = best[
            (best["tag"] == control_tag)
            & (best["postprocess"] == pp)
        ]

        r = best[
            (best["tag"] == rank_tag)
            & (best["postprocess"] == pp)
        ]

        if len(c) and len(r):
            c = c.iloc[0]
            r = r.iloc[0]

            for metric in [
                "map50",
                "map50_95",
                "precision",
                "recall",
                "postprocess_ms",
            ]:
                if metric in c.index and metric in r.index:
                    try:
                        delta[
                            f"{pp}_{metric}_delta"
                        ] = (
                            float(r[metric])
                            - float(c[metric])
                        )
                    except Exception:
                        pass

    if len(assignment):

        c = assignment[
            assignment["tag"] == control_tag
        ]

        r = assignment[
            assignment["tag"] == rank_tag
        ]

        if len(c) and len(r):
            c = c.iloc[0]
            r = r.iloc[0]

            for metric in [
                "selected_std_iou_mean",
                "selected_pred_class_correct_rate",
                "selected_gtclass_wins_iou50_rate",
                "selected_maxscore_wins_iou50_rate",
                "selected_minus_best_comp_gtclass_median",
                "selected_minus_best_comp_maxscore_median",
                "same_scale_iou50_competitors_mean",
                "cross_scale_iou50_competitors_mean",
            ]:
                if metric in c.index and metric in r.index:
                    delta[
                        f"{metric}_delta"
                    ] = (
                        float(r[metric])
                        - float(c[metric])
                    )

    if len(box):

        bx = box.copy()

        if "conf_thres" in bx.columns:
            bx["conf_thres"] = pd.to_numeric(
                bx["conf_thres"],
                errors="coerce",
            )

        mode_name = "one2one_current_no_nms"

        for tag, prefix in [
            (control_tag, "control"),
            (rank_tag, "rank"),
        ]:
            s = bx[
                (bx["tag"] == tag)
                & (bx["mode"] == mode_name)
                & (np.isclose(bx["conf_thres"], 0.01))
            ]

            if len(s):
                row = s.iloc[0]

                delta[
                    f"{prefix}_duplicate_pairs_conf001"
                ] = float(
                    row[
                        "total_duplicate_pairs_iou70_same_class"
                    ]
                )

                delta[
                    f"{prefix}_class_match_iou50_conf001"
                ] = float(
                    row[
                        "total_class_match_iou50"
                    ]
                )

        if (
            "control_duplicate_pairs_conf001" in delta
            and "rank_duplicate_pairs_conf001" in delta
        ):
            delta[
                "duplicate_pairs_conf001_delta"
            ] = (
                delta["rank_duplicate_pairs_conf001"]
                - delta["control_duplicate_pairs_conf001"]
            )

    delta_df = pd.DataFrame([delta])

    delta_df.to_csv(
        report_dir /
        "week22_controlled_delta.csv",
        index=False,
    )

    gate = pd.DataFrame([
        {
            "gate_step": 1,
            "status": "PASS_WEEK20_21",
            "evidence":
                "Same-scale high-IoU competitors and weak selected-positive score dominance confirmed.",
        },
        {
            "gate_step": 2,
            "status": "PASS_REPRESENTATIVE_DISCRIMINATION_PRINCIPLE",
            "evidence":
                "Week21 showed selected positives are frequently outranked; cross-scale competition is secondary.",
        },
        {
            "gate_step": 3,
            "status": "IMPLEMENTED_MINIMAL_YOLOV9_NATIVE",
            "evidence":
                "No new head/TAL replacement/decode change; only O2O ranking auxiliary loss.",
        },
        {
            "gate_step": 4,
            "status": "CONTROLLED_ABLATION_DESIGNED",
            "evidence":
                "Same B100 last.pt and same 20e continuation; only ranking-loss weight differs.",
        },
    ])

    gate.to_csv(
        report_dir /
        "week22_external_method_gate.csv",
        index=False,
    )

    lines = [
        "# Week22 — YOLOv9-native Representative Ranking",
        "",
        "## Training",
        "",
        train.to_markdown(index=False),
        "",
        "## Best E2E",
        "",
        best.to_markdown(index=False)
        if len(best)
        else "No E2E results.",
        "",
        "## Assignment / uniqueness",
        "",
        assignment.to_markdown(index=False)
        if len(assignment)
        else "No assignment audit.",
        "",
        "## Controlled delta",
        "",
        delta_df.to_markdown(index=False),
        "",
        "## Four-stage gate",
        "",
        gate.to_markdown(index=False),
        "",
        "## Decision criteria",
        "",
        "- Primary: selected-positive winner rates increase.",
        "- Score gaps should move toward zero/positive.",
        "- no-NMS AP should improve and duplicates should decrease.",
        "- NMS AP/localization should not materially degrade.",
        "- If direct ranking metrics do not improve, inspect detail CSV before expanding the method.",
    ]

    summary_path = (
        report_dir /
        "week22_summary.md"
    )

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    evidence = {
        "variants": list(config["variants"].keys()),
        "num_e2e_rows": int(len(sweeps)),
        "num_assignment_rows": int(len(assignment)),
        "num_box_rows": int(len(box)),
        "num_rank_telemetry_rows": int(len(telemetry)),
    }

    (
        report_dir /
        "week22_evidence.json"
    ).write_text(
        json.dumps(evidence, indent=2),
        encoding="utf-8",
    )

    print(
        summary_path.read_text(
            encoding="utf-8"
        )
    )


if __name__ == "__main__":
    main()
