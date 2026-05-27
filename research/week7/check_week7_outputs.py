from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    required = [
        "week5_week7_combined_comparison.csv",
        "week7_accuracy_table.csv",
        "week7_latency_table.csv",
        "week7_proposed_postprocess_comparison.csv",
        "week7_postprocess_speedup_summary.csv",
        "week7_extended_tables.md",
        "week7_table_summary.json",
    ]

    for name in required:
        path = out_dir / name
        assert path.exists(), f"missing: {path}"
        print("OK:", path)

    df = pd.read_csv(out_dir / "week5_week7_combined_comparison.csv")
    assert len(df) >= 3, "comparison table should have at least three rows"

    for col in ["precision", "recall", "map50", "map50_95"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            assert ((s >= 0.0) & (s <= 1.0)).all(), f"{col} out of range"

    for col in ["preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            assert (s >= 0).all(), f"{col} contains negative values"

    proposed = df[df["model"].astype(str).str.contains("proposed", case=False, na=False)]
    assert {"no-nms", "nms"}.issubset(set(proposed["postprocess"].dropna().astype(str))), "missing proposed nms/no-nms rows"

    print("Week 7 output sanity check passed")


if __name__ == "__main__":
    main()
