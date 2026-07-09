
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    args = ap.parse_args()
    root = Path(args.drive)

    required = [
        "week17_experiment_config.json",
        "selected_week16_reference_weights.json",
        "sweeps/week17_one2one_copy_long100_val_e2e_sweep.csv",
        "sweeps/week17_one2one_copy_lowaug100_val_e2e_sweep.csv",
        "prediction_probe/week16_one2one_copy_prediction_probe_aggregate.csv",
        "prediction_probe/week17_one2one_copy_long100_prediction_probe_aggregate.csv",
        "prediction_probe/week17_one2one_copy_lowaug100_prediction_probe_aggregate.csv",
        "loss_assigner_audit/loss_assigner_keyword_hits.csv",
        "loss_assigner_audit/loss_assigner_contexts.md",
        "loss_assigner_audit/loss_assigner_audit_summary.json",
        "week17_diagnosis_report/week17_diagnosis_summary.md",
        "week17_diagnosis_report/week17_best_by_tag.csv",
        "week17_diagnosis_report/week17_all_sweeps_with_week16_reference.csv",
        "week17_diagnosis_report/week17_prediction_probe_aggregate_all.csv",
        "week17_diagnosis_report/week17_raw_one2one_aggregate_all.csv",
        "week17_diagnosis_report/week17_train_summary.csv",
        "week17_diagnosis_report/week17_evidence.json",
        "runs_backup",
        "backup_files",
    ]

    for rel in required:
        p = root / rel
        assert p.exists(), f"missing: {p}"
        print("OK:", p)

    for name in [
        "week17_one2one_copy_long100_val_e2e_sweep.csv",
        "week17_one2one_copy_lowaug100_val_e2e_sweep.csv",
    ]:
        df = pd.read_csv(root / "sweeps" / name)
        assert len(df) >= 16, f"too few sweep rows: {name}"
        assert {"nms", "no-nms"}.issubset(set(df["postprocess"].dropna())), f"missing postprocess: {name}"

    best = pd.read_csv(root / "week17_diagnosis_report/week17_best_by_tag.csv")
    assert len(best) >= 3, "best table too small"
    print("Week 17 output sanity check passed")

if __name__ == "__main__":
    main()
