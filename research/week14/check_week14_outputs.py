
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    args = ap.parse_args()
    root = Path(args.drive)

    required = [
        "week14_experiment_config.json",
        "selected_week13_weights_for_week14.json",
        "source_audit/source_keyword_hits.csv",
        "source_audit/source_contexts.md",
        "source_audit/source_audit_summary.json",
        "branch_probe/branch_prediction_probe.csv",
        "branch_probe/branch_prediction_probe_aggregate.csv",
        "branch_probe/raw_output_structure.json",
        "sweeps/week13_proposed_original_val_e2e_sweep.csv",
        "sweeps/proposed_from_baseline_val_e2e_sweep.csv",
        "sweeps/proposed_continue_val_e2e_sweep.csv",
        "week14_diagnosis_report/week14_diagnosis_summary.md",
        "week14_diagnosis_report/week14_evidence.json",
        "runs_backup",
        "backup_files",
    ]

    for rel in required:
        p = root / rel
        assert p.exists(), f"missing: {p}"
        print("OK:", p)

    assert len(pd.read_csv(root / "source_audit/source_keyword_hits.csv")) > 0
    assert len(pd.read_csv(root / "branch_probe/branch_prediction_probe_aggregate.csv")) > 0

    for name in [
        "week13_proposed_original_val_e2e_sweep.csv",
        "proposed_from_baseline_val_e2e_sweep.csv",
        "proposed_continue_val_e2e_sweep.csv",
    ]:
        df = pd.read_csv(root / "sweeps" / name)
        assert len(df) >= 10, f"too few rows: {name}"
        assert {"nms", "no-nms"}.issubset(set(df["postprocess"].dropna())), name

    print("Week 14 output sanity check passed")

if __name__ == "__main__":
    main()
