
import argparse
import json
from pathlib import Path
import pandas as pd


def read_csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def metric_col(df, aliases):
    for a in aliases:
        if a in df.columns:
            return a
    return None


def summarize_training(run_dir, tag):
    results = Path(run_dir) / "results.csv"
    if not results.exists():
        return {"tag": tag, "error": f"missing {results}"}

    df = pd.read_csv(results)
    row = {
        "tag": tag,
        "epochs_recorded": int(len(df)),
        "results_path": str(results),
    }

    aliases = {
        "precision": ["metrics/precision", "metrics/precision(B)", "precision"],
        "recall": ["metrics/recall", "metrics/recall(B)", "recall"],
        "map50": ["metrics/mAP_0.5", "metrics/mAP50(B)", "map50"],
        "map50_95": ["metrics/mAP_0.5:0.95", "metrics/mAP50-95(B)", "map50_95"],
    }

    for out_name, cand in aliases.items():
        c = metric_col(df, cand)
        if c is not None:
            s = pd.to_numeric(df[c], errors="coerce")
            row[f"final_{out_name}"] = float(s.iloc[-1]) if len(s) else None
            row[f"best_{out_name}"] = float(s.max()) if s.notna().any() else None

    return row


def best_e2e(df):
    rows = []

    if len(df) == 0:
        return pd.DataFrame()

    tmp = df.copy()

    for c in ["precision", "recall", "map50", "map50_95", "postprocess_ms", "conf_thres"]:
        if c in tmp.columns:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    for tag in sorted(tmp["tag"].dropna().unique()):
        for pp in ["nms", "no-nms"]:
            sub = tmp[
                (tmp["tag"] == tag)
                & (tmp["postprocess"] == pp)
            ].dropna(subset=["map50"])

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
        (root / "week19p2_experiment_config.json").read_text(encoding="utf-8")
    )

    # Training summaries
    train_rows = []
    for tag, spec in config["variants"].items():
        run_dir = root / "runs_backup" / spec["run_name"]
        train_rows.append(summarize_training(run_dir, tag))

    train_df = pd.DataFrame(train_rows)
    train_df.to_csv(
        report_dir / "week19p2_training_summary.csv",
        index=False,
    )

    # E2E sweep
    sweep_tables = []
    for p in sorted((root / "e2e_sweeps").glob("*_val_e2e_sweep.csv")):
        d = pd.read_csv(p)
        d["source_file"] = p.name
        sweep_tables.append(d)

    sweeps = (
        pd.concat(sweep_tables, ignore_index=True)
        if sweep_tables else pd.DataFrame()
    )
    sweeps.to_csv(
        report_dir / "week19p2_all_e2e_sweeps.csv",
        index=False,
    )

    best = best_e2e(sweeps)
    best.to_csv(
        report_dir / "week19p2_best_e2e_by_variant.csv",
        index=False,
    )

    # Box probe
    probe_tables = []
    for p in sorted((root / "box_probe").glob("*_nonms_probe_aggregate.csv")):
        d = pd.read_csv(p)
        d["source_file"] = p.name
        probe_tables.append(d)

    probe = (
        pd.concat(probe_tables, ignore_index=True)
        if probe_tables else pd.DataFrame()
    )
    probe.to_csv(
        report_dir / "week19p2_box_probe_aggregate.csv",
        index=False,
    )

    # Score margin
    margin_tables = []
    for p in sorted((root / "score_margin").glob("*_score_margin_summary.csv")):
        margin_tables.append(pd.read_csv(p))

    margin = (
        pd.concat(margin_tables, ignore_index=True)
        if margin_tables else pd.DataFrame()
    )
    margin.to_csv(
        report_dir / "week19p2_score_margin_summary.csv",
        index=False,
    )

    # Telemetry
    telemetry_rows = []
    for tag in config["variants"]:
        p = root / "telemetry" / tag / "loss_telemetry_summary.csv"
        if not p.exists():
            continue

        d = pd.read_csv(p)

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

        row = {"tag": tag}

        for metric in wanted:
            m = d[d["metric"] == metric]
            if len(m):
                row[metric + "_mean"] = float(m.iloc[0]["mean"])
                row[metric + "_last"] = float(m.iloc[0]["last"])

        telemetry_rows.append(row)

    telemetry = pd.DataFrame(telemetry_rows)
    telemetry.to_csv(
        report_dir / "week19p2_telemetry_summary.csv",
        index=False,
    )

    # A/B/C/D delta table — best E2E based
    delta_rows = []

    if len(best):
        pivot = {}

        for tag in config["variants"]:
            pivot[tag] = {}

            for pp in ["nms", "no-nms"]:
                s = best[
                    (best["tag"] == tag)
                    & (best["postprocess"] == pp)
                ]

                if len(s):
                    pivot[tag][pp] = s.iloc[0]

        comparisons = [
            ("A_to_B_same_feature", "A_current_split_w025", "B_samefeat_w025"),
            ("B_to_C_copy_init", "B_samefeat_w025", "C_samefeat_copy_w025"),
            ("C_to_D_o2m_weight", "C_samefeat_copy_w025", "D_samefeat_copy_w100"),
        ]

        for label, base, test in comparisons:
            row = {
                "comparison": label,
                "base": base,
                "test": test,
            }

            for pp in ["nms", "no-nms"]:
                if pp in pivot.get(base, {}) and pp in pivot.get(test, {}):
                    b = pivot[base][pp]
                    t = pivot[test][pp]

                    for metric in ["map50", "map50_95", "precision", "recall", "postprocess_ms"]:
                        if metric in b.index and metric in t.index:
                            try:
                                row[f"{pp}_{metric}_delta"] = (
                                    float(t[metric]) - float(b[metric])
                                )
                            except Exception:
                                pass

            delta_rows.append(row)

    delta = pd.DataFrame(delta_rows)
    delta.to_csv(
        report_dir / "week19p2_controlled_deltas.csv",
        index=False,
    )

    # Markdown summary
    lines = []
    lines.append("# Week19 Part2 Controlled Screening Summary")
    lines.append("")
    lines.append("## Training summary")
    lines.append("")
    lines.append(train_df.to_markdown(index=False) if len(train_df) else "No training summary.")
    lines.append("")
    lines.append("## Best E2E metrics by variant / postprocess")
    lines.append("")
    lines.append(best.to_markdown(index=False) if len(best) else "No E2E results.")
    lines.append("")
    lines.append("## Controlled deltas")
    lines.append("")
    lines.append(delta.to_markdown(index=False) if len(delta) else "No delta table.")
    lines.append("")
    lines.append("## Score margin summary")
    lines.append("")
    lines.append(margin.to_markdown(index=False) if len(margin) else "No score-margin results.")
    lines.append("")
    lines.append("## Telemetry summary")
    lines.append("")
    lines.append(telemetry.to_markdown(index=False) if len(telemetry) else "No telemetry.")
    lines.append("")
    lines.append("## Interpretation rule")
    lines.append("")
    lines.append("- A→B is the primary same-feature causal comparison.")
    lines.append("- B→C isolates copy initialization.")
    lines.append("- C→D isolates O2M loss-weight alignment.")
    lines.append("- 10 epochs is screening only; do not treat it as final performance.")
    lines.append("- Duplicate/margin results determine whether DeFCN/PSS-style uniqueness learning is needed next.")

    summary_path = report_dir / "week19p2_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    evidence = {
        "variants": list(config["variants"].keys()),
        "num_training_rows": int(len(train_df)),
        "num_sweep_rows": int(len(sweeps)),
        "num_best_rows": int(len(best)),
        "num_probe_rows": int(len(probe)),
        "num_margin_rows": int(len(margin)),
        "num_telemetry_rows": int(len(telemetry)),
        "num_delta_rows": int(len(delta)),
    }

    (report_dir / "week19p2_evidence.json").write_text(
        json.dumps(evidence, indent=2),
        encoding="utf-8",
    )

    print(summary_path.read_text(encoding="utf-8"))
    print("\nEvidence:")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
