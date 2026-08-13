
import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED = {
    "A_split_w025",
    "B_samefeat_w025",
    "C_samefeat_copy_w025",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    args = ap.parse_args()

    root = Path(args.drive)

    config = json.loads(
        (
            root / "week20_experiment_config.json"
        ).read_text(encoding="utf-8")
    )

    assert set(config["variants"]) == EXPECTED

    required = [
        "source_audit/week20_existing_source_audit.json",
        "source_audit/official_v10postprocess_audit.json",
        "source_audit/official_v10postprocess_excerpt.txt",
        "reports/week20_training_summary.csv",
        "reports/week20_all_threeway_e2e_sweeps.csv",
        "reports/week20_best_e2e_by_variant.csv",
        "reports/week20_controlled_deltas_last.csv",
        "reports/week20_box_probe_aggregate.csv",
        "reports/week20_refined_ranking_summary.csv",
        "reports/week20_telemetry_summary.csv",
        "reports/week20_external_method_gate.csv",
        "reports/week20_summary.md",
        "reports/week20_evidence.json",
    ]

    for rel in required:
        p = root / rel
        assert p.exists(), f"missing: {p}"
        print("OK:", p)

    for tag, spec in config["variants"].items():
        run_dir = (
            root /
            "runs_backup" /
            spec["run_name"]
        )

        results = run_dir / "results.csv"
        last = run_dir / "weights/last.pt"
        best = run_dir / "weights/best.pt"

        assert results.exists(), results
        assert last.exists(), last
        assert best.exists(), best
        assert last.stat().st_size > 1_000_000
        assert best.stat().st_size > 1_000_000

        df = pd.read_csv(results)

        assert len(df) >= config["epochs"], (
            f"{tag}: expected {config['epochs']} epochs, got {len(df)}"
        )

    sweeps = pd.read_csv(
        root /
        "reports/week20_all_threeway_e2e_sweeps.csv"
    )

    assert EXPECTED.issubset(set(sweeps["tag"]))
    assert {"last", "best"}.issubset(
        set(sweeps["weight_kind"])
    )
    assert {
        "nms",
        "no-nms",
        "official-v10",
    }.issubset(
        set(sweeps["postprocess"])
    )

    expected_rows = (
        len(EXPECTED)
        * 2
        * 3
        * len(config["conf_sweep"])
    )

    assert len(sweeps) >= expected_rows, (
        f"expected >= {expected_rows}, got {len(sweeps)}"
    )

    errors = sweeps["error"].fillna("").astype(str)

    assert (errors == "").all(), (
        sweeps[errors != ""].to_string()
    )

    box = pd.read_csv(
        root /
        "reports/week20_box_probe_aggregate.csv"
    )

    assert {
        "native_nms",
        "one2one_nms",
        "one2one_current_no_nms",
        "one2one_official_v10",
    }.issubset(set(box["mode"]))

    rank = pd.read_csv(
        root /
        "reports/week20_refined_ranking_summary.csv"
    )

    assert EXPECTED.issubset(set(rank["tag"]))

    rounded_iou = {
        round(float(x), 3)
        for x in rank["iou_threshold"]
    }

    assert {0.3, 0.5}.issubset(rounded_iou)

    tel = pd.read_csv(
        root /
        "reports/week20_telemetry_summary.csv"
    )

    assert EXPECTED.issubset(set(tel["tag"]))

    delta = pd.read_csv(
        root /
        "reports/week20_controlled_deltas_last.csv"
    )

    assert {
        "A_to_B_same_feature",
        "B_to_C_copy_init",
    }.issubset(set(delta["comparison"]))

    print("\nWeek20 output sanity check passed.")


if __name__ == "__main__":
    main()
