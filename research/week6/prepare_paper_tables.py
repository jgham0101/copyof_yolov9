from __future__ import annotations

from pathlib import Path
import argparse
import json
import pandas as pd


REQUIRED_COLUMNS = [
    "model",
    "postprocess",
    "precision",
    "recall",
    "map50",
    "map50_95",
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",
]


def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def to_markdown(df: pd.DataFrame, path: Path):
    path.write_text(df.to_markdown(index=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--week5-csv", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--repeat-csv", type=str, default="")
    args = parser.parse_args()

    week5_csv = Path(args.week5_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not week5_csv.exists():
        raise FileNotFoundError(f"missing week5 csv: {week5_csv}")

    df = pd.read_csv(week5_csv)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"missing required column: {col}")

    numeric_cols = [
        "precision",
        "recall",
        "map50",
        "map50_95",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["total_ms"] = df[["preprocess_ms", "inference_ms", "postprocess_ms"]].sum(axis=1, skipna=False)

    # Accuracy table
    acc_cols = ["model", "postprocess", "precision", "recall", "map50", "map50_95"]
    accuracy = df[acc_cols].copy()
    accuracy.to_csv(out_dir / "accuracy_table.csv", index=False)
    to_markdown(accuracy, out_dir / "accuracy_table.md")

    # Latency table
    lat_cols = ["model", "postprocess", "preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"]
    latency = df[lat_cols].copy()
    latency.to_csv(out_dir / "latency_table.csv", index=False)
    to_markdown(latency, out_dir / "latency_table.md")

    # Proposed postprocess comparison
    proposed = df[df["model"].astype(str).str.contains("proposed", case=False, na=False)].copy()
    post_comp = proposed[["model", "postprocess", "map50", "map50_95", "postprocess_ms", "total_ms"]].copy()

    no_nms = proposed[proposed["postprocess"].astype(str).str.contains("no", case=False, na=False)]
    nms = proposed[
        proposed["postprocess"].astype(str).str.lower().eq("nms")
        | proposed["postprocess"].astype(str).str.contains("nms", case=False, na=False)
    ]

    no_nms = no_nms[~no_nms["postprocess"].astype(str).str.contains("^nms$", case=False, regex=True)]
    nms = proposed[proposed["postprocess"].astype(str).str.lower() == "nms"]

    speedup = None
    if len(no_nms) > 0 and len(nms) > 0:
        no_nms_ms = safe_float(no_nms.iloc[-1]["postprocess_ms"])
        nms_ms = safe_float(nms.iloc[-1]["postprocess_ms"])
        if no_nms_ms and no_nms_ms > 0 and nms_ms is not None:
            speedup = nms_ms / no_nms_ms

    post_comp.to_csv(out_dir / "postprocess_comparison.csv", index=False)
    to_markdown(post_comp, out_dir / "postprocess_comparison.md")

    repeat_summary = None
    if args.repeat_csv:
        repeat_path = Path(args.repeat_csv)
        if repeat_path.exists():
            repeat = pd.read_csv(repeat_path)
            repeat_numeric = ["precision", "recall", "map50", "map50_95", "preprocess_ms", "inference_ms", "postprocess_ms"]
            for col in repeat_numeric:
                if col in repeat.columns:
                    repeat[col] = pd.to_numeric(repeat[col], errors="coerce")

            group_cols = ["postprocess"]
            agg_cols = [col for col in repeat_numeric if col in repeat.columns]
            repeat_summary = repeat.groupby(group_cols)[agg_cols].agg(["mean", "std"]).reset_index()
            repeat_summary.to_csv(out_dir / "repeat_eval_summary.csv", index=False)

    summary = {
        "week5_csv": str(week5_csv),
        "rows": len(df),
        "models": sorted(df["model"].dropna().astype(str).unique().tolist()),
        "postprocesses": sorted(df["postprocess"].dropna().astype(str).unique().tolist()),
        "proposed_postprocess_speedup_nms_over_no_nms": speedup,
        "outputs": {
            "accuracy_table": str(out_dir / "accuracy_table.csv"),
            "latency_table": str(out_dir / "latency_table.csv"),
            "postprocess_comparison": str(out_dir / "postprocess_comparison.csv"),
        },
    }

    (out_dir / "paper_tables_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = []
    md_lines.append("# Week 6 Paper Tables\\n")
    md_lines.append("## Accuracy Table\\n")
    md_lines.append(accuracy.to_markdown(index=False))
    md_lines.append("\\n\\n## Latency Table\\n")
    md_lines.append(latency.to_markdown(index=False))
    md_lines.append("\\n\\n## Proposed Postprocess Comparison\\n")
    md_lines.append(post_comp.to_markdown(index=False))
    md_lines.append("\\n\\n## Interpretation Notes\\n")
    md_lines.append("- Current results are smoke-scale COCO128 / 1 epoch results.\\n")
    md_lines.append("- mAP values may be very low or zero; they should not be interpreted as final paper performance.\\n")
    md_lines.append("- The reliable Week 6 output is the comparison pipeline and table format.\\n")
    if speedup is not None:
        md_lines.append(f"- Proposed postprocess speedup, NMS over no-NMS: {speedup:.4f}x.\\n")
    else:
        md_lines.append("- Proposed postprocess speedup could not be computed due to missing values.\\n")

    (out_dir / "week6_paper_tables.md").write_text("\\n".join(md_lines), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
