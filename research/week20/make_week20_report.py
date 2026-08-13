
import argparse
import json
from pathlib import Path

import pandas as pd


def read_csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def summarize_train(run_dir, tag):
    p = Path(run_dir) / "results.csv"

    if not p.exists():
        return {
            "tag": tag,
            "error": f"missing {p}",
        }

    df = pd.read_csv(p)

    row = {
        "tag": tag,
        "epochs_recorded": int(len(df)),
        "results_path": str(p),
    }

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

        if s.notna().any():
            row[f"best_{name}_epoch"] = int(
                s.idxmax()
            )

    return row


def best_by_group(df):
    rows = []

    if len(df) == 0:
        return pd.DataFrame()

    tmp = df.copy()

    for c in [
        "precision",
        "recall",
        "map50",
        "map50_95",
        "postprocess_ms",
        "conf_thres",
    ]:
        if c in tmp.columns:
            tmp[c] = pd.to_numeric(
                tmp[c],
                errors="coerce",
            )

    for _, sub in tmp.groupby(
        [
            "tag",
            "weight_kind",
            "postprocess",
        ]
    ):
        sub = sub.dropna(
            subset=["map50"]
        )

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
            "week20_experiment_config.json"
        ).read_text(encoding="utf-8")
    )

    train_rows = []

    for tag, spec in config["variants"].items():
        run_dir = (
            root /
            "runs_backup" /
            spec["run_name"]
        )
        train_rows.append(
            summarize_train(run_dir, tag)
        )

    train = pd.DataFrame(train_rows)
    train.to_csv(
        report_dir / "week20_training_summary.csv",
        index=False,
    )

    sweep_tables = []

    for d in [
        root / "e2e_sweeps_last",
        root / "e2e_sweeps_best",
    ]:
        for p in sorted(
            d.glob("*_threeway_e2e_sweep.csv")
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
        report_dir / "week20_all_threeway_e2e_sweeps.csv",
        index=False,
    )

    best = best_by_group(sweeps)

    best.to_csv(
        report_dir / "week20_best_e2e_by_variant.csv",
        index=False,
    )

    last_best = (
        best[best["weight_kind"] == "last"].copy()
        if len(best)
        else pd.DataFrame()
    )

    comparisons = [
        (
            "A_to_B_same_feature",
            "A_split_w025",
            "B_samefeat_w025",
        ),
        (
            "B_to_C_copy_init",
            "B_samefeat_w025",
            "C_samefeat_copy_w025",
        ),
    ]

    causal_rows = []

    for label, base, test in comparisons:
        row = {
            "comparison": label,
            "base": base,
            "test": test,
        }

        for pp in [
            "nms",
            "no-nms",
            "official-v10",
        ]:
            b = last_best[
                (last_best["tag"] == base)
                & (last_best["postprocess"] == pp)
            ]
            t = last_best[
                (last_best["tag"] == test)
                & (last_best["postprocess"] == pp)
            ]

            if len(b) and len(t):
                b = b.iloc[0]
                t = t.iloc[0]

                for metric in [
                    "precision",
                    "recall",
                    "map50",
                    "map50_95",
                    "postprocess_ms",
                ]:
                    try:
                        row[
                            f"{pp}_{metric}_delta"
                        ] = (
                            float(t[metric])
                            - float(b[metric])
                        )
                    except Exception:
                        pass

        causal_rows.append(row)

    causal = pd.DataFrame(causal_rows)

    causal.to_csv(
        report_dir / "week20_controlled_deltas_last.csv",
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
        report_dir / "week20_box_probe_aggregate.csv",
        index=False,
    )

    rank_tables = []

    for p in sorted(
        (root / "ranking_probe").glob("*_refined_ranking_summary.csv")
    ):
        rank_tables.append(pd.read_csv(p))

    ranking = (
        pd.concat(rank_tables, ignore_index=True)
        if rank_tables
        else pd.DataFrame()
    )

    ranking.to_csv(
        report_dir / "week20_refined_ranking_summary.csv",
        index=False,
    )

    tel_rows = []

    wanted = [
        "fg_mask_sum_o2m",
        "fg_mask_sum_o2o",
        "target_scores_sum_o2m",
        "target_scores_sum_o2o",
        "loss_cls_o2m_raw",
        "loss_cls_o2o_raw",
        "loss_box_o2m_raw",
        "loss_box_o2o_raw",
        "loss_dfl_o2m_raw",
        "loss_dfl_o2o_raw",
    ]

    for tag in config["variants"]:
        p = (
            root /
            "telemetry" /
            tag /
            "loss_telemetry_summary.csv"
        )

        if not p.exists():
            continue

        d = pd.read_csv(p)
        row = {"tag": tag}

        for metric in wanted:
            m = d[d["metric"] == metric]

            if len(m):
                row[metric + "_mean"] = float(
                    m.iloc[0]["mean"]
                )
                row[metric + "_last"] = float(
                    m.iloc[0]["last"]
                )

        tel_rows.append(row)

    telemetry = pd.DataFrame(tel_rows)

    telemetry.to_csv(
        report_dir / "week20_telemetry_summary.csv",
        index=False,
    )

    ref_dir = report_dir / "week19p2_reference"

    week19_best = read_csv(
        ref_dir / "week19p2_best_e2e_by_variant.csv"
    )

    week19_best.to_csv(
        report_dir / "week20_week19p2_best_reference.csv",
        index=False,
    )

    gate = pd.DataFrame([
        {
            "gate_step": 1,
            "question":
                "Is the remaining YOLOv9 problem confirmed by Week20 data?",
            "status": "REQUIRES_INTERPRETATION",
            "evidence":
                "Use 100e localization, class-match, ranking and duplicate tables.",
        },
        {
            "gate_step": 2,
            "question":
                "Does the confirmed problem match DeFCN/PSS prior-work problem?",
            "status": "NOT_EVALUATED_YET",
            "evidence":
                "Evaluate only after Gate 1.",
        },
        {
            "gate_step": 3,
            "question":
                "Can the principle be redesigned minimally inside YOLOv9?",
            "status": "NOT_EVALUATED_YET",
            "evidence":
                "No module transplantation in Week20.",
        },
        {
            "gate_step": 4,
            "question":
                "Can the change be isolated by one-factor ablation?",
            "status": "NOT_EVALUATED_YET",
            "evidence":
                "Design only after Gates 1-3.",
        },
    ])

    gate.to_csv(
        report_dir / "week20_external_method_gate.csv",
        index=False,
    )

    lines = [
        "# Week20 Long-Convergence / Official Top-k Summary",
        "",
        "## Training convergence",
        "",
        train.to_markdown(index=False) if len(train) else "No training summary.",
        "",
        "## Best three-way E2E metrics",
        "",
        best.to_markdown(index=False) if len(best) else "No E2E results.",
        "",
        "## Fixed-horizon controlled deltas (last.pt)",
        "",
        causal.to_markdown(index=False) if len(causal) else "No controlled delta table.",
        "",
        "## Four-mode box probe",
        "",
        box.to_markdown(index=False) if len(box) else "No box probe.",
        "",
        "## Refined ranking probe",
        "",
        ranking.to_markdown(index=False) if len(ranking) else "No ranking probe.",
        "",
        "## Telemetry",
        "",
        telemetry.to_markdown(index=False) if len(telemetry) else "No telemetry.",
        "",
        "## External-method four-stage gate",
        "",
        gate.to_markdown(index=False),
        "",
        "## Interpretation rules",
        "",
        "- A→B isolates same-feature under full 100-epoch convergence.",
        "- B→C isolates O2O copy initialization.",
        "- current no-NMS vs official-v10 isolates postprocess ranking.",
        "- If localization is healthy but GT-class ranking/class-match remains weak, assignment/ranking becomes the next target.",
        "- DeFCN/PSS must not be added until the four-stage gate is reviewed.",
    ]

    summary_path = report_dir / "week20_summary.md"
    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    evidence = {
        "num_training_variants": int(len(train)),
        "num_sweep_rows": int(len(sweeps)),
        "num_best_rows": int(len(best)),
        "num_box_probe_rows": int(len(box)),
        "num_ranking_rows": int(len(ranking)),
        "num_telemetry_rows": int(len(telemetry)),
        "num_controlled_comparisons": int(len(causal)),
    }

    (
        report_dir / "week20_evidence.json"
    ).write_text(
        json.dumps(evidence, indent=2),
        encoding="utf-8",
    )

    print(
        summary_path.read_text(encoding="utf-8")
    )


if __name__ == "__main__":
    main()
