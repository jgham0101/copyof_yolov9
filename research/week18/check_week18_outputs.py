
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    args = ap.parse_args()

    root = Path(args.drive)
    required = [
        "week18_experiment_config.json",
        "selected_week17_reference_weights.json",
        "nonms_probe/week17_lowaug100_nonms_probe_aggregate.csv",
        "nonms_probe/week17_lowaug100_nonms_probe_detail.csv",
        "nonms_probe/week17_lowaug100_output_shapes.csv",
        "loss_telemetry/loss_telemetry.jsonl",
        "loss_telemetry/loss_telemetry_detail.csv",
        "loss_telemetry/loss_telemetry_summary.csv",
        "loss_telemetry/loss_telemetry_ratios.csv",
        "week18_diagnosis_report/week18_diagnosis_summary.md",
        "week18_diagnosis_report/week18_nonms_probe_aggregate_all.csv",
        "week18_diagnosis_report/week18_loss_telemetry_summary.csv",
        "week18_diagnosis_report/week18_loss_telemetry_ratios.csv",
        "week18_diagnosis_report/week18_evidence.json",
        "runs_backup",
        "backup_files",
    ]

    for rel in required:
        p = root / rel
        assert p.exists(), f"missing: {p}"
        print("OK:", p)

    probe = pd.read_csv(root / "nonms_probe/week17_lowaug100_nonms_probe_aggregate.csv")
    assert "one2one_no_nms" in set(probe["mode"].dropna()), "one2one_no_nms rows missing"
    no_nms = probe[probe["mode"] == "one2one_no_nms"]
    assert len(no_nms) > 0, "empty no-NMS probe"
    assert no_nms["errors"].fillna(0).sum() == 0, "no-NMS probe errors remain"

    tel = pd.read_csv(root / "loss_telemetry/loss_telemetry_detail.csv")
    assert len(tel) > 0, "empty telemetry"
    for col in ["fg_mask_sum_o2m", "fg_mask_sum_o2o", "target_scores_sum_o2m", "target_scores_sum_o2o"]:
        assert col in tel.columns, f"missing telemetry column: {col}"

    print("Week 18 output sanity check passed")

if __name__ == "__main__":
    main()
