
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
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(
        (
            root /
            "week21_experiment_config.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    summary_tables = []
    detail_tables = []

    for tag in config["variants"]:
        s = (
            root /
            "assignment_probe" /
            f"{tag}_assignment_uniqueness_summary.csv"
        )

        d = (
            root /
            "assignment_probe" /
            f"{tag}_assignment_uniqueness_detail.csv"
        )

        assert s.exists(), s
        assert d.exists(), d

        summary_tables.append(
            pd.read_csv(s)
        )

        x = pd.read_csv(d)
        x["source_tag"] = tag
        detail_tables.append(x)

    summary = pd.concat(
        summary_tables,
        ignore_index=True,
    )

    detail = pd.concat(
        detail_tables,
        ignore_index=True,
    )

    summary.to_csv(
        report_dir /
        "week21_assignment_uniqueness_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Diagnostic classification tables
    # --------------------------------------------------------

    diagnostic_rows = []

    for tag, d in detail.groupby("tag"):

        selected = d["selected_after_conflict"] == 1

        selected_good_box = (
            selected
            & (
                pd.to_numeric(
                    d["selected_std_iou"],
                    errors="coerce",
                ).fillna(0)
                >= 0.5
            )
        )

        multi_candidate = (
            d["iou50_candidate_count"] >= 2
        )

        selected_good_multi = (
            selected_good_box
            & multi_candidate
        )

        row = {
            "tag": tag,
            "num_gt": len(d),

            "gt_with_iou50_candidate_rate":
                float(
                    (
                        d["iou50_candidate_count"] > 0
                    ).mean()
                ),

            "gt_with_multiple_iou50_candidates_rate":
                float(
                    multi_candidate.mean()
                ),

            "selected_good_box_rate":
                float(
                    selected_good_box.mean()
                ),

            "selected_good_multi_candidate_rate":
                float(
                    selected_good_multi.mean()
                ),

            "selected_class_correct_rate_on_good_box":
                float(
                    d.loc[
                        selected_good_box,
                        "selected_pred_class_correct",
                    ].mean()
                )
                if selected_good_box.any()
                else 0.0,

            "selected_wins_gtclass_rate_on_good_multi":
                float(
                    d.loc[
                        selected_good_multi,
                        "selected_gtclass_outscores_iou50_competitors",
                    ].mean()
                )
                if selected_good_multi.any()
                else 0.0,

            "selected_wins_maxscore_rate_on_good_multi":
                float(
                    d.loc[
                        selected_good_multi,
                        "selected_maxscore_outscores_iou50_competitors",
                    ].mean()
                )
                if selected_good_multi.any()
                else 0.0,

            "selected_matches_inference_gtclass_rate_on_good_multi":
                float(
                    d.loc[
                        selected_good_multi,
                        "selected_matches_inference_gtclass_iou50",
                    ].mean()
                )
                if selected_good_multi.any()
                else 0.0,

            "selected_matches_inference_maxscore_rate_on_good_multi":
                float(
                    d.loc[
                        selected_good_multi,
                        "selected_matches_inference_maxscore_iou50",
                    ].mean()
                )
                if selected_good_multi.any()
                else 0.0,

            "cross_scale_duplicate_context_rate_on_multi":
                float(
                    d.loc[
                        multi_candidate,
                        "iou50_has_cross_scale_candidates",
                    ].mean()
                )
                if multi_candidate.any()
                else 0.0,

            "same_scale_competitors_mean_on_good_multi":
                float(
                    d.loc[
                        selected_good_multi,
                        "same_scale_iou50_competitors",
                    ].mean()
                )
                if selected_good_multi.any()
                else 0.0,

            "cross_scale_competitors_mean_on_good_multi":
                float(
                    d.loc[
                        selected_good_multi,
                        "cross_scale_iou50_competitors",
                    ].mean()
                )
                if selected_good_multi.any()
                else 0.0,

            "selected_minus_comp_gtclass_median_on_good_multi":
                float(
                    d.loc[
                        selected_good_multi,
                        "selected_minus_best_comp_gtclass",
                    ].median()
                )
                if selected_good_multi.any()
                else 0.0,

            "selected_minus_comp_maxscore_median_on_good_multi":
                float(
                    d.loc[
                        selected_good_multi,
                        "selected_minus_best_comp_maxscore",
                    ].median()
                )
                if selected_good_multi.any()
                else 0.0,
        }

        diagnostic_rows.append(row)

    diagnostic = pd.DataFrame(
        diagnostic_rows
    )

    diagnostic.to_csv(
        report_dir /
        "week21_diagnostic_decision_table.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Scale table
    # --------------------------------------------------------

    scale_rows = []

    for tag, d in detail.groupby("tag"):
        selected = d[
            d["selected_after_conflict"] == 1
        ]

        for level in range(3):
            scale_rows.append({
                "tag": tag,
                "level": level,
                "scale": f"P{level + 3}",
                "selected_count": int(
                    (
                        selected["selected_level"]
                        == level
                    ).sum()
                ),
                "selected_rate": float(
                    (
                        selected["selected_level"]
                        == level
                    ).mean()
                ) if len(selected) else 0.0,
                "iou50_candidates_total": int(
                    d[
                        f"iou50_level{level}_count"
                    ].sum()
                ),
                "iou50_candidates_mean_per_gt":
                    float(
                        d[
                            f"iou50_level{level}_count"
                        ].mean()
                    ),
            })

    scale = pd.DataFrame(scale_rows)

    scale.to_csv(
        report_dir /
        "week21_scale_competition_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Gate worksheet — no automatic PASS for Gate 2+
    # --------------------------------------------------------

    gate = pd.DataFrame([
        {
            "gate_step": 1,
            "name": "Problem confirmed in YOLOv9",
            "status": "PASS_FROM_WEEK20",
            "question":
                "Does 100e YOLOv9 O2O retain multiple high-IoU candidates and weak score separation?",
            "week21_evidence":
                "Reconfirm with assignment-selected vs inference competitor statistics.",
        },
        {
            "gate_step": 2,
            "name": "Problem matches prior work",
            "status": "REQUIRES_INTERPRETATION",
            "question":
                "Is failure mainly assignment choice, runner-up suppression, or cross-scale discriminability?",
            "week21_evidence":
                "Use week21_diagnostic_decision_table.csv and scale competition table.",
        },
        {
            "gate_step": 3,
            "name": "YOLOv9-native minimal redesign",
            "status": "NOT_STARTED",
            "question":
                "Can the matching/uniqueness principle be implemented inside current TAL/head with minimal change?",
            "week21_evidence":
                "Do not design until Gate 2 is interpreted.",
        },
        {
            "gate_step": 4,
            "name": "Single-variable ablation",
            "status": "NOT_STARTED",
            "question":
                "Can the proposed change be isolated against B same-feature baseline?",
            "week21_evidence":
                "Design only after Gate 3.",
        },
    ])

    gate.to_csv(
        report_dir /
        "week21_external_method_gate.csv",
        index=False,
    )

    lines = []

    lines.append(
        "# Week21 — O2O Assignment & Uniqueness Contract Audit"
    )
    lines.append("")

    lines.append("## Assignment summary")
    lines.append("")
    lines.append(
        summary.to_markdown(index=False)
    )
    lines.append("")

    lines.append("## Diagnostic decision table")
    lines.append("")
    lines.append(
        diagnostic.to_markdown(index=False)
    )
    lines.append("")

    lines.append("## Scale competition")
    lines.append("")
    lines.append(
        scale.to_markdown(index=False)
    )
    lines.append("")

    lines.append("## Four-stage external-method gate")
    lines.append("")
    lines.append(
        gate.to_markdown(index=False)
    )
    lines.append("")

    lines.append("## Interpretation logic")
    lines.append("")

    lines.append(
        "- If TAL-selected positives frequently have low IoU or disagree with best-IoU / best-GT-class candidates, assignment formulation is the first suspect."
    )

    lines.append(
        "- If TAL-selected positives are good boxes but are often outranked by IoU>=0.5 competitors, the main problem is representative discrimination / runner-up suppression rather than basic localization."
    )

    lines.append(
        "- If most competitor sets span multiple feature scales, cross-scale discriminability becomes a justified investigation target."
    )

    lines.append(
        "- DeFCN/PSS code is not transplanted in Week21. Only the observed failure mode determines which prior-work principle may be relevant."
    )

    summary_path = (
        report_dir /
        "week21_summary.md"
    )

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    evidence = {
        "variants": list(
            config["variants"].keys()
        ),
        "num_gt_rows": int(
            len(detail)
        ),
        "num_summary_rows": int(
            len(summary)
        ),
        "num_diagnostic_rows": int(
            len(diagnostic)
        ),
        "num_scale_rows": int(
            len(scale)
        ),
    }

    (
        report_dir /
        "week21_evidence.json"
    ).write_text(
        json.dumps(
            evidence,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        summary_path.read_text(
            encoding="utf-8"
        )
    )


if __name__ == "__main__":
    main()
