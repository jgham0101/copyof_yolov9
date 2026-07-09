
import argparse
import json
from pathlib import Path
import pandas as pd

def read_csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def best_by_tag(df):
    if len(df) == 0:
        return pd.DataFrame()
    tmp = df.copy()
    for c in ["precision", "recall", "map50", "map50_95", "postprocess_ms"]:
        if c in tmp.columns:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
    out = []
    for tag in sorted(tmp["tag"].dropna().unique()):
        sub = tmp[tmp["tag"] == tag].dropna(subset=["map50"])
        if len(sub):
            out.append(sub.sort_values("map50", ascending=False).iloc[0].to_dict())
    return pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week16-sweeps", required=True)
    ap.add_argument("--week17-sweep-dir", required=True)
    ap.add_argument("--train-runs-dir", required=True)
    ap.add_argument("--prediction-probe-dir", required=True)
    ap.add_argument("--source-audit-summary", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    week16 = read_csv(args.week16_sweeps)

    sweep_tables = []
    for f in sorted(Path(args.week17_sweep_dir).glob("*_val_e2e_sweep.csv")):
        df = pd.read_csv(f)
        df["sweep_file"] = f.name
        sweep_tables.append(df)
    week17 = pd.concat(sweep_tables, ignore_index=True) if sweep_tables else pd.DataFrame()

    combined = pd.concat([week16, week17], ignore_index=True, sort=False) if len(week16) else week17
    combined.to_csv(out_dir / "week17_all_sweeps_with_week16_reference.csv", index=False)
    best = best_by_tag(combined)
    best.to_csv(out_dir / "week17_best_by_tag.csv", index=False)

    train_rows = []
    for run in sorted(Path(args.train_runs_dir).glob("week17_*")):
        results = run / "results.csv"
        if not results.exists():
            continue
        df = pd.read_csv(results)
        numeric = df.copy()
        for c in numeric.columns:
            numeric[c] = pd.to_numeric(numeric[c], errors="ignore")
        row = {"run": run.name, "rows": len(df), "results_path": str(results)}
        for col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any():
                row[f"first_{col}"] = s.dropna().iloc[0]
                row[f"last_{col}"] = s.dropna().iloc[-1]
                row[f"min_{col}"] = s.min()
                row[f"max_{col}"] = s.max()
        train_rows.append(row)
    train_summary = pd.DataFrame(train_rows)
    train_summary.to_csv(out_dir / "week17_train_summary.csv", index=False)

    probe_tables = []
    raw_tables = []
    for f in sorted(Path(args.prediction_probe_dir).glob("*_prediction_probe_aggregate.csv")):
        probe_tables.append(pd.read_csv(f))
    for f in sorted(Path(args.prediction_probe_dir).glob("*_raw_one2one_aggregate.csv")):
        raw_tables.append(pd.read_csv(f))
    probe_agg = pd.concat(probe_tables, ignore_index=True) if probe_tables else pd.DataFrame()
    raw_agg = pd.concat(raw_tables, ignore_index=True) if raw_tables else pd.DataFrame()
    probe_agg.to_csv(out_dir / "week17_prediction_probe_aggregate_all.csv", index=False)
    raw_agg.to_csv(out_dir / "week17_raw_one2one_aggregate_all.csv", index=False)

    source_summary = {}
    p = Path(args.source_audit_summary)
    if p.exists():
        source_summary = json.loads(p.read_text(encoding="utf-8"))

    lines = []
    lines.append("# Week 17 COCO128 Convergence / no-NMS / Assigner Diagnosis Summary")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Week 17 checks three issues from Week 16: COCO128 convergence, no-NMS score/top-k behavior, and loss/assigner implementation evidence.")
    lines.append("")
    lines.append("## Best E2E results by tag")
    lines.append("")
    lines.append(best.to_markdown(index=False) if len(best) else "No best table.")
    lines.append("")
    lines.append("## Week 17 train summary")
    lines.append("")
    lines.append(train_summary.to_markdown(index=False) if len(train_summary) else "No train summary.")
    lines.append("")
    lines.append("## Prediction probe aggregate")
    lines.append("")
    lines.append(probe_agg.to_markdown(index=False) if len(probe_agg) else "No prediction probe aggregate.")
    lines.append("")
    lines.append("## Raw one2one aggregate")
    lines.append("")
    lines.append(raw_agg.to_markdown(index=False) if len(raw_agg) else "No raw one2one aggregate.")
    lines.append("")
    lines.append("## Source audit summary")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(source_summary, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("- If longer training improves NMS but not no-NMS, branch score ranking/top-k remains the key issue.")
    lines.append("- If low-augmentation training improves both NMS and no-NMS, COCO128 convergence/overfit difficulty was a major bottleneck.")
    lines.append("- If no-NMS has many candidates but low AP, score ranking or selection quality is weak.")
    lines.append("- If one2one raw confidence remains very low, confidence calibration or branch supervision is the next target.")
    lines.append("")

    (out_dir / "week17_diagnosis_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "week17_evidence.json").write_text(json.dumps({
        "num_week16_rows": int(len(week16)),
        "num_week17_rows": int(len(week17)),
        "num_combined_rows": int(len(combined)),
        "num_train_runs": int(len(train_summary)),
        "num_probe_rows": int(len(probe_agg)),
        "source_summary": source_summary,
    }, indent=2), encoding="utf-8")

    print("\n".join(lines))

if __name__ == "__main__":
    main()
