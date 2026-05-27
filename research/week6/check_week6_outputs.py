from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    required_files = [
        "accuracy_table.csv",
        "latency_table.csv",
        "postprocess_comparison.csv",
        "week6_paper_tables.md",
        "paper_tables_summary.json",
    ]

    for name in required_files:
        path = out_dir / name
        assert path.exists(), f"missing file: {path}"
        print("OK:", path)

    acc = pd.read_csv(out_dir / "accuracy_table.csv")
    lat = pd.read_csv(out_dir / "latency_table.csv")
    post = pd.read_csv(out_dir / "postprocess_comparison.csv")

    for df, name in [(acc, "accuracy"), (lat, "latency"), (post, "postprocess")]:
        assert len(df) > 0, f"{name} table is empty"

    for col in ["precision", "recall", "map50", "map50_95"]:
        if col in acc.columns:
            valid = pd.to_numeric(acc[col], errors="coerce").dropna()
            assert ((valid >= 0.0) & (valid <= 1.0)).all(), f"{col} out of range"

    for col in ["preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"]:
        if col in lat.columns:
            valid = pd.to_numeric(lat[col], errors="coerce").dropna()
            assert (valid >= 0.0).all(), f"{col} contains negative values"

    print("Week 6 output sanity check passed")


if __name__ == "__main__":
    main()
