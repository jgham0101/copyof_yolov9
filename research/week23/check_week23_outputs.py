
import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED = {
    "control_continue20",
    "pairwise_continue20",
    "positive_only_continue20",
    "negative_only_continue20",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    args = ap.parse_args()

    root = Path(args.drive)

    config = json.loads(
        (
            root /
            "week23_experiment_config.json"
        ).read_text(encoding="utf-8")
    )

    assert set(config["all_variants"]) == EXPECTED

    required = [
        "source_audit/week23_source_audit.json",
        "source_audit/week23_patch_audit.json",
        "reports/week23_all_e2e_sweeps.csv",
        "reports/week23_best_e2e_by_variant.csv",
        "reports/week23_fixed_conf001_e2e.csv",
        "reports/week23_assignment_uniqueness_summary.csv",
        "reports/week23_common_gt_mechanism_summary.csv",
        "reports/week23_paired_mechanism_deltas.csv",
        "reports/week23_box_probe_aggregate.csv",
        "reports/week23_fixed_conf001_box_mechanism.csv",
        "reports/week23_controlled_mechanism_deltas.csv",
        "reports/week23_gradient_mode_telemetry_summary.csv",
        "reports/week23_mechanism_decision_gate.csv",
        "reports/week23_summary.md",
    ]

    for rel in required:
        p = root / rel
        assert p.exists(), p
        print("OK:", p)

    for tag, spec in config["train_variants"].items():
        run = (
            root /
            "runs_backup" /
            spec["run_name"]
        )

        results = run / "results.csv"
        last = run / "weights/last.pt"
        best = run / "weights/best.pt"

        assert results.exists()
        assert last.exists()
        assert best.exists()

        df = pd.read_csv(results)
        assert len(df) >= config["epochs"]

    sweeps = pd.read_csv(
        root /
        "reports/week23_all_e2e_sweeps.csv"
    )

    assert EXPECTED.issubset(
        set(sweeps["tag"])
    )

    assert {"nms", "no-nms"}.issubset(
        set(sweeps["postprocess"])
    )

    errors = sweeps["error"].fillna("").astype(str)
    assert (errors == "").all(), (
        sweeps[errors != ""].to_string()
    )

    common = pd.read_csv(
        root /
        "reports/week23_common_gt_mechanism_summary.csv"
    )

    assert EXPECTED.issubset(
        set(common["tag"])
    )

    assert common["num_common_gt"].min() > 0

    paired = pd.read_csv(
        root /
        "reports/week23_paired_mechanism_deltas.csv"
    )

    assert len(paired) == 3

    box = pd.read_csv(
        root /
        "reports/week23_fixed_conf001_box_mechanism.csv"
    )

    assert EXPECTED.issubset(
        set(box["tag"])
    )

    delta = pd.read_csv(
        root /
        "reports/week23_controlled_mechanism_deltas.csv"
    )

    assert len(delta) == 3

    print("\nWeek23 output sanity check passed.")


if __name__ == "__main__":
    main()
