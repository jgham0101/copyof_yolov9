
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    args = ap.parse_args()

    root = Path(args.drive)

    structure_path = (
        root / "gradient_probe/week19_structure_smoke.csv"
    )
    grad_path = (
        root / "gradient_probe/week19_gradient_path_probe.csv"
    )
    audit_path = (
        root / "source_audit/source_audit_before_week19.json"
    )

    assert structure_path.exists(), structure_path
    assert grad_path.exists(), grad_path
    assert audit_path.exists(), audit_path

    structure = pd.read_csv(structure_path)
    grad = pd.read_csv(grad_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    pivot = grad.pivot(
        index="variant",
        columns="group",
        values="grad_norm",
    )

    structure.to_csv(
        report_dir / "week19_structure_smoke.csv",
        index=False,
    )

    grad.to_csv(
        report_dir / "week19_gradient_path_probe.csv",
        index=False,
    )

    pivot.to_csv(
        report_dir / "week19_gradient_norm_pivot.csv"
    )

    lines = []

    lines.append(
        "# Week 19 Same-feature / Gradient Path Alignment Summary"
    )
    lines.append("")

    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "Week19 compares the current split-feature V10DualDDetect "
        "with same-feature variants before adding DeFCN/PSS-style changes."
    )
    lines.append("")

    lines.append("## Source audit")
    lines.append("")
    checks = audit.get("checks", {})
    for key, value in checks.items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    lines.append("## Structure smoke")
    lines.append("")
    lines.append(structure.to_markdown(index=False))
    lines.append("")

    lines.append("## Gradient norm pivot")
    lines.append("")
    lines.append(pivot.to_markdown())
    lines.append("")

    lines.append("## Interpretation rule")
    lines.append("")
    lines.append(
        "- Do not infer causality from a single gradient norm."
    )
    lines.append(
        "- Compare A→B first to isolate the same-feature effect."
    )
    lines.append(
        "- Compare B→C next to isolate O2O copy initialization."
    )
    lines.append(
        "- Compare C→D next to isolate O2M loss-weight alignment."
    )
    lines.append(
        "- DeFCN/PSS-style assignment/selector changes are deferred "
        "until the basic pipeline is mechanically verified."
    )
    lines.append("")

    summary_path = (
        report_dir / "week19_alignment_summary.md"
    )

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    evidence = {
        "base_commit": audit.get("base_commit"),
        "week19_branch": audit.get("week19_branch"),
        "num_source_checks": len(checks),
        "num_source_checks_true": int(
            sum(bool(v) for v in checks.values())
        ),
        "num_structure_rows": int(len(structure)),
        "num_gradient_rows": int(len(grad)),
        "variants_structure": sorted(
            structure["variant"].dropna().unique().tolist()
        ),
        "variants_gradient": sorted(
            grad["variant"].dropna().unique().tolist()
        ),
        "groups_gradient": sorted(
            grad["group"].dropna().unique().tolist()
        ),
    }

    (
        report_dir / "week19_evidence.json"
    ).write_text(
        json.dumps(evidence, indent=2),
        encoding="utf-8",
    )

    print(summary_path.read_text(encoding="utf-8"))
    print("\nEvidence:")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
