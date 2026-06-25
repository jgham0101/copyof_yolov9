
import argparse, json
from pathlib import Path
import pandas as pd

def read_csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week13-val", required=True)
    ap.add_argument("--week13-loss", required=True)
    ap.add_argument("--source-audit", required=True)
    ap.add_argument("--branch-probe", required=True)
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    week13_val = read_csv(args.week13_val)
    week13_loss = read_csv(args.week13_loss)
    source_summary = json.loads(Path(args.source_audit).read_text(encoding="utf-8"))
    branch_probe = read_csv(args.branch_probe)

    sweep_files = sorted(Path(args.sweep_dir).glob("*_val_e2e_sweep.csv"))
    sweeps = []
    for f in sweep_files:
        df = pd.read_csv(f)
        df["sweep_file"] = f.name
        sweeps.append(df)
    sweep = pd.concat(sweeps, ignore_index=True) if sweeps else pd.DataFrame()

    lines = [
        "# Week 14 Loss / Assigner / Branch Diagnosis Summary",
        "",
        "## Week 13 reference validation",
        "",
        week13_val.to_markdown(index=False),
        "",
        "## Week 13 reference loss",
        "",
        week13_loss.to_markdown(index=False),
        "",
        "## Source audit summary",
        "",
        f"- Scanned files: {source_summary.get('num_files_scanned')}",
        f"- Keyword hits: {source_summary.get('num_keyword_hits')}",
        f"- Context blocks: {source_summary.get('num_contexts')}",
        "",
        "## Branch output probe aggregate",
        "",
        branch_probe.to_markdown(index=False) if len(branch_probe) else "No branch probe table.",
        "",
        "## Week 14 E2E sweep results",
        "",
    ]

    if len(sweep):
        cols = [c for c in ["tag", "postprocess", "conf_thres", "precision", "recall", "map50", "map50_95", "postprocess_ms", "error"] if c in sweep.columns]
        lines.append(sweep[cols].to_markdown(index=False))
    else:
        lines.append("No sweep results.")

    lines += [
        "",
        "## Diagnostic guide",
        "",
        "- If `proposed_from_baseline` is much better than Week 13 proposed, initialization / recipe is a major factor.",
        "- If `proposed_continue` improves strongly, training duration / convergence is a major factor.",
        "- If neither improves enough, loss / assigner / branch supervision mismatch remains the dominant hypothesis.",
        "- If one2one NMS and no-NMS remain similar, NMS removal is still not the primary cause.",
        "",
    ]

    if len(sweep) and "map50" in sweep.columns:
        lines += ["## Automatic best-by-tag", ""]
        tmp = sweep.copy()
        tmp["map50"] = pd.to_numeric(tmp["map50"], errors="coerce")
        for tag in sorted(tmp["tag"].dropna().unique()):
            sub = tmp[tmp["tag"] == tag].dropna(subset=["map50"])
            if len(sub):
                best = sub.sort_values("map50", ascending=False).iloc[0]
                lines.append(f"- {tag}: best {best['postprocess']} conf={best['conf_thres']} mAP50={best['map50']} mAP50-95={best.get('map50_95')}")

    (out_dir / "week14_diagnosis_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "week14_evidence.json").write_text(json.dumps({
        "sweep_files": [str(f) for f in sweep_files],
        "num_sweep_rows": int(len(sweep)),
        "branch_probe_rows": int(len(branch_probe)),
        "source_summary": source_summary,
    }, indent=2), encoding="utf-8")
    if len(sweep):
        sweep.to_csv(out_dir / "week14_all_sweeps.csv", index=False)
    print("\n".join(lines))

if __name__ == "__main__":
    main()
