
from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    args = ap.parse_args()

    root = Path(args.drive)

    required = [
        "week11_experiment_config.json",
        "selected_weights.json",
        "branch_output_structure.json",
        "v10dual_forward_source.txt",
        "week11_branch_metric_comparison.csv",
        "week11_branch_metric_comparison.json",
        "week11_diagnosis_report/week11_branch_delta_comparison.csv",
        "week11_diagnosis_report/week11_diagnosis_summary.md",
        "week11_diagnosis_report/week11_diagnosis_evidence.json",
        "logs/week11_baseline_reference_val_dual.log",
        "logs/week11_proposed_native_val_dual.log",
    ]

    for rel in required:
        p = root / rel
        assert p.exists(), f"missing: {p}"
        print("OK:", p)

    df = pd.read_csv(root / "week11_branch_metric_comparison.csv")
    assert len(df) >= 4, "expected at least 4 rows"

    for col in ["precision", "recall", "map50", "map50_95"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            assert ((s >= 0) & (s <= 1)).all(), f"{col} out of range"

    for col in ["preprocess_ms", "inference_ms", "postprocess_ms"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            assert (s >= 0).all(), f"{col} negative"

    labels = set(df["label"].dropna().astype(str))
    assert "baseline_nms_reference" in labels
    assert "proposed_native_val_dual" in labels
    assert "proposed_one2one_nms" in labels
    assert "proposed_one2one_no_nms" in labels

    print("Week 11 output sanity check passed")


if __name__ == "__main__":
    main()
