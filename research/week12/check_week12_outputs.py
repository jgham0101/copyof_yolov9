
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
        "week12_experiment_config.json",
        "selected_weights.json",
        "sample_manifest.json",
        "prediction_analysis/prediction_count_confidence_summary.csv",
        "prediction_analysis/prediction_aggregate_summary.csv",
        "prediction_analysis/prediction_errors.json",
        "threshold_sweep/threshold_sweep_results.csv",
        "week12_diagnosis_report/week12_diagnosis_summary.md",
        "week12_diagnosis_report/week12_evidence.json",
    ]

    for rel in required:
        p = root / rel
        assert p.exists(), f"missing: {p}"
        print("OK:", p)

    pred = pd.read_csv(root / "prediction_analysis/prediction_count_confidence_summary.csv")
    agg = pd.read_csv(root / "prediction_analysis/prediction_aggregate_summary.csv")
    sweep = pd.read_csv(root / "threshold_sweep/threshold_sweep_results.csv")

    assert len(pred) > 0, "empty prediction summary"
    assert len(agg) >= 2, "too few prediction modes"
    assert len(sweep) >= 4, "too few threshold sweep rows"

    labels = set(agg["label"].astype(str))
    expected_any = {"baseline_native_nms", "proposed_one2one_nms", "proposed_one2one_no_nms"}
    assert expected_any.intersection(labels), f"expected labels not found: {labels}"

    print("Week 12 output sanity check passed")


if __name__ == "__main__":
    main()
