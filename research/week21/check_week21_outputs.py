
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED = {
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
            root /
            "week21_experiment_config.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert set(
        config["variants"]
    ) == EXPECTED

    required = [
        "source_audit/week21_source_contract_audit.json",
        "source_audit/week21_assigner_context.txt",
        "reports/week21_assignment_uniqueness_summary.csv",
        "reports/week21_diagnostic_decision_table.csv",
        "reports/week21_scale_competition_summary.csv",
        "reports/week21_external_method_gate.csv",
        "reports/week21_summary.md",
        "reports/week21_evidence.json",
    ]

    for rel in required:
        p = root / rel
        assert p.exists(), p
        print("OK:", p)

    for tag in EXPECTED:
        detail = (
            root /
            "assignment_probe" /
            f"{tag}_assignment_uniqueness_detail.csv"
        )

        summary = (
            root /
            "assignment_probe" /
            f"{tag}_assignment_uniqueness_summary.csv"
        )

        assert detail.exists(), detail
        assert summary.exists(), summary

        d = pd.read_csv(detail)

        assert len(d) >= 900, (
            f"{tag}: expected about COCO128 GT count, got {len(d)}"
        )

        required_cols = [
            "align_gap",
            "selected_after_conflict",
            "selected_std_iou",
            "iou50_candidate_count",
            "selected_matches_inference_gtclass_iou50",
            "selected_minus_best_comp_gtclass",
            "same_scale_iou50_competitors",
            "cross_scale_iou50_competitors",
        ]

        for col in required_cols:
            assert col in d.columns, col

        numeric = pd.to_numeric(
            d["align_gap"],
            errors="coerce",
        )

        assert np.isfinite(
            numeric.fillna(0)
        ).all()

    summary = pd.read_csv(
        root /
        "reports/week21_assignment_uniqueness_summary.csv"
    )

    assert EXPECTED.issubset(
        set(summary["tag"])
    )

    diagnostic = pd.read_csv(
        root /
        "reports/week21_diagnostic_decision_table.csv"
    )

    assert EXPECTED.issubset(
        set(diagnostic["tag"])
    )

    gate = pd.read_csv(
        root /
        "reports/week21_external_method_gate.csv"
    )

    assert set(gate["gate_step"]) == {
        1, 2, 3, 4
    }

    print(
        "\nWeek21 output sanity check passed."
    )


if __name__ == "__main__":
    main()
