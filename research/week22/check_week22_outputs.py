
import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED = {
    "control_continue20",
    "rank_continue20",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    args = ap.parse_args()

    root = Path(args.drive)

    config = json.loads(
        (
            root /
            "week22_experiment_config.json"
        ).read_text(encoding="utf-8")
    )

    assert set(config["variants"]) == EXPECTED

    required = [
        "source_audit/week22_source_audit.json",
        "source_audit/week22_patch_audit.json",
        "reports/week22_training_summary.csv",
        "reports/week22_all_e2e_sweeps.csv",
        "reports/week22_best_e2e_by_variant.csv",
        "reports/week22_assignment_uniqueness_summary.csv",
        "reports/week22_box_probe_aggregate.csv",
        "reports/week22_rank_telemetry.csv",
        "reports/week22_controlled_delta.csv",
        "reports/week22_external_method_gate.csv",
        "reports/week22_summary.md",
        "reports/week22_evidence.json",
    ]

    for rel in required:
        p = root / rel
        assert p.exists(), p
        print("OK:", p)

    for tag, spec in config["variants"].items():
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

        assert len(df) >= config["epochs"], (
            tag,
            len(df),
        )

    sweeps = pd.read_csv(
        root /
        "reports/week22_all_e2e_sweeps.csv"
    )

    assert EXPECTED.issubset(set(sweeps["tag"]))
    assert {"nms", "no-nms"}.issubset(
        set(sweeps["postprocess"])
    )

    errors = (
        sweeps["error"]
        .fillna("")
        .astype(str)
    )

    assert (errors == "").all(), (
        sweeps[errors != ""].to_string()
    )

    assignment = pd.read_csv(
        root /
        "reports/week22_assignment_uniqueness_summary.csv"
    )

    assert EXPECTED.issubset(
        set(assignment["tag"])
    )

    telemetry = pd.read_csv(
        root /
        "reports/week22_rank_telemetry.csv"
    )

    assert len(telemetry) > 0

    assert (
        pd.to_numeric(
            telemetry["active_positive_count"],
            errors="coerce",
        )
        .fillna(0)
        .max()
        > 0
    )

    delta = pd.read_csv(
        root /
        "reports/week22_controlled_delta.csv"
    )

    assert len(delta) == 1

    gate = pd.read_csv(
        root /
        "reports/week22_external_method_gate.csv"
    )

    assert set(gate["gate_step"]) == {1, 2, 3, 4}

    print("\nWeek22 output sanity check passed.")


if __name__ == "__main__":
    main()
