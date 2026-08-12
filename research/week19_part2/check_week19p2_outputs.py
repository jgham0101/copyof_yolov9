
import argparse
import json
from pathlib import Path
import pandas as pd


EXPECTED = {
    "A_current_split_w025",
    "B_samefeat_w025",
    "C_samefeat_copy_w025",
    "D_samefeat_copy_w100",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    args = ap.parse_args()

    root = Path(args.drive)
    config = json.loads(
        (root / "week19p2_experiment_config.json").read_text(encoding="utf-8")
    )

    assert set(config["variants"]) == EXPECTED

    required = [
        "source_audit/week19p2_source_audit.json",
        "reports/week19p2_training_summary.csv",
        "reports/week19p2_all_e2e_sweeps.csv",
        "reports/week19p2_best_e2e_by_variant.csv",
        "reports/week19p2_box_probe_aggregate.csv",
        "reports/week19p2_score_margin_summary.csv",
        "reports/week19p2_telemetry_summary.csv",
        "reports/week19p2_controlled_deltas.csv",
        "reports/week19p2_summary.md",
        "reports/week19p2_evidence.json",
    ]

    for rel in required:
        p = root / rel
        assert p.exists(), f"missing: {p}"
        print("OK:", p)

    # Training run integrity
    for tag, spec in config["variants"].items():
        run = root / "runs_backup" / spec["run_name"]
        results = run / "results.csv"
        last = run / "weights/last.pt"
        best = run / "weights/best.pt"

        assert results.exists(), results
        assert last.exists() and last.stat().st_size > 1_000_000, last
        assert best.exists() and best.stat().st_size > 1_000_000, best

        df = pd.read_csv(results)
        assert len(df) >= config["epochs"], (
            f"{tag}: expected >= {config['epochs']} epochs, got {len(df)}"
        )

    # E2E sweep completeness
    sweeps = pd.read_csv(root / "reports/week19p2_all_e2e_sweeps.csv")
    assert EXPECTED.issubset(set(sweeps["tag"]))

    expected_sweep_rows = (
        len(EXPECTED)
        * 2
        * len(config["conf_sweep"])
    )
    assert len(sweeps) >= expected_sweep_rows, (
        f"Expected at least {expected_sweep_rows} sweep rows, got {len(sweeps)}"
    )

    if "error" in sweeps.columns:
        errors = sweeps["error"].fillna("").astype(str)
        assert (errors == "").all(), (
            "E2E sweep contains errors:\n"
            + sweeps[errors != ""].to_string()
        )

    # Margin completeness
    margin = pd.read_csv(root / "reports/week19p2_score_margin_summary.csv")
    assert EXPECTED.issubset(set(margin["tag"]))

    # Telemetry completeness
    tel = pd.read_csv(root / "reports/week19p2_telemetry_summary.csv")
    assert EXPECTED.issubset(set(tel["tag"]))

    # Controlled delta rows
    delta = pd.read_csv(root / "reports/week19p2_controlled_deltas.csv")
    expected_cmp = {
        "A_to_B_same_feature",
        "B_to_C_copy_init",
        "C_to_D_o2m_weight",
    }
    assert expected_cmp.issubset(set(delta["comparison"]))

    print("\nWeek19 Part2 output sanity check passed")


if __name__ == "__main__":
    main()
