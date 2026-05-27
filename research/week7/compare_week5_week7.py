from __future__ import annotations

from pathlib import Path
import argparse
import json
import pandas as pd


def load_csv(path: Path, label: str):
    if not path.exists():
        print(f"missing {label}: {path}")
        return None
    df = pd.read_csv(path)
    df["experiment"] = label
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--week5", type=str, required=True)
    parser.add_argument("--week7", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    frames = []
    for path, label in [(Path(args.week5), "week5_e1"), (Path(args.week7), "week7_e3")]:
        df = load_csv(path, label)
        if df is not None:
            frames.append(df)

    if not frames:
        raise FileNotFoundError("No comparison CSV files found")

    df = pd.concat(frames, ignore_index=True)

    numeric = ["precision", "recall", "map50", "map50_95", "preprocess_ms", "inference_ms", "postprocess_ms"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if {"preprocess_ms", "inference_ms", "postprocess_ms"}.issubset(df.columns):
        df["total_ms"] = df[["preprocess_ms", "inference_ms", "postprocess_ms"]].sum(axis=1, skipna=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "week5_week7_combined_comparison.csv", index=False)

    acc = df[[c for c in ["experiment", "model", "postprocess", "precision", "recall", "map50", "map50_95"] if c in df.columns]]
    lat = df[[c for c in ["experiment", "model", "postprocess", "preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"] if c in df.columns]]
    prop = df[df["model"].astype(str).str.contains("proposed", case=False, na=False)].copy()

    acc.to_csv(out_dir / "week7_accuracy_table.csv", index=False)
    lat.to_csv(out_dir / "week7_latency_table.csv", index=False)
    prop.to_csv(out_dir / "week7_proposed_postprocess_comparison.csv", index=False)

    rows = []
    for exp, g in prop.groupby("experiment"):
        no = g[g["postprocess"].astype(str).str.contains("no", case=False, na=False)]
        nms = g[g["postprocess"].astype(str).str.lower() == "nms"]
        row = {"experiment": exp}
        if len(no) and len(nms):
            no_ms = pd.to_numeric(no.iloc[-1]["postprocess_ms"], errors="coerce")
            nms_ms = pd.to_numeric(nms.iloc[-1]["postprocess_ms"], errors="coerce")
            row["no_nms_postprocess_ms"] = float(no_ms) if pd.notna(no_ms) else None
            row["nms_postprocess_ms"] = float(nms_ms) if pd.notna(nms_ms) else None
            if row["no_nms_postprocess_ms"] and row["no_nms_postprocess_ms"] > 0 and row["nms_postprocess_ms"] is not None:
                row["speedup_nms_over_no_nms"] = row["nms_postprocess_ms"] / row["no_nms_postprocess_ms"]
            else:
                row["speedup_nms_over_no_nms"] = None
        rows.append(row)

    speedup = pd.DataFrame(rows)
    speedup.to_csv(out_dir / "week7_postprocess_speedup_summary.csv", index=False)

    md = []
    md.append("# Week 7 Extended Experiment Tables\n")
    md.append("## Accuracy Table\n")
    md.append(acc.to_markdown(index=False))
    md.append("\n\n## Latency Table\n")
    md.append(lat.to_markdown(index=False))
    md.append("\n\n## Proposed Postprocess Comparison\n")
    md.append(prop.to_markdown(index=False))
    md.append("\n\n## Postprocess Speedup Summary\n")
    md.append(speedup.to_markdown(index=False))
    (out_dir / "week7_extended_tables.md").write_text("\n".join(md), encoding="utf-8")

    summary = {
        "rows": len(df),
        "experiments": sorted(df["experiment"].dropna().unique().tolist()),
        "outputs": [
            "week5_week7_combined_comparison.csv",
            "week7_accuracy_table.csv",
            "week7_latency_table.csv",
            "week7_proposed_postprocess_comparison.csv",
            "week7_postprocess_speedup_summary.csv",
            "week7_extended_tables.md",
        ],
    }
    (out_dir / "week7_table_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(df)
    print(speedup)
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
