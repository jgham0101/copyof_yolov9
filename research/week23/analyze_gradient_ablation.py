
import argparse
from pathlib import Path

import pandas as pd


VARIANTS = [
    "control_continue20",
    "pairwise_continue20",
    "positive_only_continue20",
    "negative_only_continue20",
]

KEY = [
    "image",
    "gt_local_index",
    "gt_class",
]


def load_detail(root, tag):
    p = (
        root /
        "assignment_probe" /
        f"{tag}_assignment_uniqueness_detail.csv"
    )

    assert p.exists(), p

    df = pd.read_csv(p)

    assert not df.duplicated(KEY).any(), (
        f"duplicate GT key in {tag}"
    )

    return df


def eligible(df):
    return (
        (df["selected_after_conflict"] == 1)
        &
        (
            pd.to_numeric(
                df["selected_std_iou"],
                errors="coerce",
            ).fillna(0)
            >= 0.5
        )
        &
        (
            pd.to_numeric(
                df["same_scale_iou50_competitors"],
                errors="coerce",
            ).fillna(0)
            > 0
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    args = ap.parse_args()

    root = Path(args.drive)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    tables = {
        tag: load_detail(root, tag)
        for tag in VARIANTS
    }

    key_sets = []

    for tag, df in tables.items():
        keys = set(
            map(
                tuple,
                df[KEY].to_numpy(),
            )
        )

        key_sets.append(keys)
        print(tag, "GT rows:", len(df))

    common_universe = set.intersection(*key_sets)
    assert len(common_universe) >= 900, len(common_universe)

    eligible_sets = []

    for tag, df in tables.items():
        e = df[eligible(df)]

        eligible_sets.append(
            set(
                map(
                    tuple,
                    e[KEY].to_numpy(),
                )
            )
        )

    common_eligible = set.intersection(*eligible_sets)
    assert len(common_eligible) > 0

    common_key_df = pd.DataFrame(
        list(common_eligible),
        columns=KEY,
    )

    common_key_df.to_csv(
        reports /
        "week23_common_eligible_gt_keys.csv",
        index=False,
    )

    summary_rows = []
    common_tables = {}

    for tag, df in tables.items():
        d = common_key_df.merge(
            df,
            on=KEY,
            how="left",
            validate="one_to_one",
        )

        common_tables[tag] = d

        summary_rows.append({
            "tag": tag,
            "num_common_gt": int(len(d)),
            "selected_gtclass_winner_rate":
                float(
                    d[
                        "selected_gtclass_outscores_iou50_competitors"
                    ].mean()
                ),
            "selected_maxscore_winner_rate":
                float(
                    d[
                        "selected_maxscore_outscores_iou50_competitors"
                    ].mean()
                ),
            "selected_minus_comp_gtclass_median":
                float(
                    d[
                        "selected_minus_best_comp_gtclass"
                    ].median()
                ),
            "selected_minus_comp_gtclass_mean":
                float(
                    d[
                        "selected_minus_best_comp_gtclass"
                    ].mean()
                ),
            "selected_minus_comp_maxscore_median":
                float(
                    d[
                        "selected_minus_best_comp_maxscore"
                    ].median()
                ),
            "selected_minus_comp_maxscore_mean":
                float(
                    d[
                        "selected_minus_best_comp_maxscore"
                    ].mean()
                ),
            "selected_gtclass_score_median":
                float(
                    d[
                        "selected_gtclass_score"
                    ].median()
                ),
            "best_competitor_gtclass_score_median":
                float(
                    d[
                        "best_competitor_gtclass_score"
                    ].median()
                ),
            "selected_maxscore_median":
                float(
                    d[
                        "selected_maxscore"
                    ].median()
                ),
            "best_competitor_maxscore_median":
                float(
                    d[
                        "best_competitor_maxscore"
                    ].median()
                ),
            "selected_std_iou_mean":
                float(
                    d[
                        "selected_std_iou"
                    ].mean()
                ),
            "selected_pred_class_correct_rate":
                float(
                    d[
                        "selected_pred_class_correct"
                    ].mean()
                ),
            "same_scale_competitors_mean":
                float(
                    d[
                        "same_scale_iou50_competitors"
                    ].mean()
                ),
        })

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        reports /
        "week23_common_gt_mechanism_summary.csv",
        index=False,
    )

    control = common_tables["control_continue20"]
    paired_rows = []

    for tag in [
        "pairwise_continue20",
        "positive_only_continue20",
        "negative_only_continue20",
    ]:
        treat = common_tables[tag]

        row = {
            "comparison": f"{tag}_minus_control",
            "num_common_gt": int(len(control)),
        }

        for metric in [
            "selected_minus_best_comp_gtclass",
            "selected_minus_best_comp_maxscore",
            "selected_gtclass_score",
            "best_competitor_gtclass_score",
            "selected_maxscore",
            "best_competitor_maxscore",
            "selected_std_iou",
        ]:
            diff = (
                pd.to_numeric(
                    treat[metric],
                    errors="coerce",
                )
                -
                pd.to_numeric(
                    control[metric],
                    errors="coerce",
                )
            )

            row[f"{metric}_delta_median"] = float(
                diff.median()
            )
            row[f"{metric}_delta_mean"] = float(
                diff.mean()
            )

        for metric, prefix in [
            (
                "selected_gtclass_outscores_iou50_competitors",
                "gtclass_winner",
            ),
            (
                "selected_maxscore_outscores_iou50_competitors",
                "maxscore_winner",
            ),
        ]:
            c = control[metric].astype(int)
            t = treat[metric].astype(int)

            row[f"{prefix}_rate_control"] = float(c.mean())
            row[f"{prefix}_rate_treatment"] = float(t.mean())
            row[f"{prefix}_rate_delta"] = float(
                t.mean() - c.mean()
            )
            row[f"{prefix}_lose_to_win"] = int(
                ((c == 0) & (t == 1)).sum()
            )
            row[f"{prefix}_win_to_lose"] = int(
                ((c == 1) & (t == 0)).sum()
            )

        paired_rows.append(row)

    paired = pd.DataFrame(paired_rows)

    paired.to_csv(
        reports /
        "week23_paired_mechanism_deltas.csv",
        index=False,
    )

    print("\n=== Common GT mechanism ===")
    print(summary.to_string(index=False))

    print("\n=== Paired mechanism deltas ===")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
