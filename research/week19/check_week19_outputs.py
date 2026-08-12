
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_VARIANTS = {
    "A_current_split_w025",
    "B_samefeat_w025",
    "C_samefeat_copy_w025",
    "D_samefeat_copy_w100",
}

REQUIRED_GROUPS = {
    "main_neck_total",
    "aux_neck_total",
    "o2m_head",
    "o2o_head",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    args = ap.parse_args()

    root = Path(args.drive)

    required = [
        "base_week18_commit.txt",
        "selected_base_branch.txt",
        "week19_experiment_config.json",
        "source_audit/source_audit_before_week19.json",
        "gradient_probe/week19_structure_smoke.csv",
        "gradient_probe/week19_gradient_path_probe.csv",
        "reports/week19_alignment_summary.md",
        "reports/week19_gradient_norm_pivot.csv",
        "reports/week19_evidence.json",
        "backup_files",
    ]

    for rel in required:
        p = root / rel
        assert p.exists(), f"missing: {p}"
        print("OK:", p)

    audit = json.loads(
        (
            root /
            "source_audit/source_audit_before_week19.json"
        ).read_text(encoding="utf-8")
    )

    checks = audit.get("checks", {})
    assert checks, "empty source audit"
    assert all(bool(v) for v in checks.values()), (
        "source audit contains False"
    )

    structure = pd.read_csv(
        root /
        "gradient_probe/week19_structure_smoke.csv"
    )

    assert EXPECTED_VARIANTS.issubset(
        set(structure["variant"])
    ), "missing structure variant"

    a = structure[
        structure["variant"] ==
        "A_current_split_w025"
    ].iloc[0]

    assert not (
        bool(a["same_ref_p3"])
        and bool(a["same_ref_p4"])
        and bool(a["same_ref_p5"])
    ), "Current variant unexpectedly shares all feature refs"

    for tag in [
        "B_samefeat_w025",
        "C_samefeat_copy_w025",
        "D_samefeat_copy_w100",
    ]:
        row = structure[
            structure["variant"] == tag
        ].iloc[0]

        assert (
            bool(row["same_ref_p3"])
            and bool(row["same_ref_p4"])
            and bool(row["same_ref_p5"])
        ), f"{tag}: same-feature check failed"

    for tag in [
        "C_samefeat_copy_w025",
        "D_samefeat_copy_w100",
    ]:
        row = structure[
            structure["variant"] == tag
        ].iloc[0]

        assert float(row["cv2_max_abs_diff"]) < 1e-8
        assert float(row["cv3_max_abs_diff"]) < 1e-8

    grad = pd.read_csv(
        root /
        "gradient_probe/week19_gradient_path_probe.csv"
    )

    assert EXPECTED_VARIANTS.issubset(
        set(grad["variant"])
    ), "missing gradient variant"

    assert REQUIRED_GROUPS.issubset(
        set(grad["group"])
    ), "missing gradient group"

    assert np.isfinite(
        pd.to_numeric(
            grad["grad_norm"],
            errors="coerce",
        )
    ).all(), "non-finite grad norm"

    # O2O head가 gradient를 실제로 받는다는 최소 조건만 강제.
    # main/aux neck의 상대적 gradient 패턴은 연구 결과이므로 assert하지 않음.
    for tag in EXPECTED_VARIANTS:
        row = grad[
            (grad["variant"] == tag)
            & (grad["group"] == "o2o_head")
        ]
        assert len(row) == 1, f"{tag}: o2o_head row missing"
        assert float(row.iloc[0]["grad_norm"]) > 0, (
            f"{tag}: O2O head has no gradient"
        )

    print("\nWeek 19 output sanity check passed")


if __name__ == "__main__":
    main()
