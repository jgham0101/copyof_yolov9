from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--dataset-root", required=True)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    ds = Path(args.dataset_root)

    assert csv_path.exists(), f"missing csv: {csv_path}"
    assert ds.exists(), f"missing dataset root: {ds}"

    for rel in [
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
        "subset_manifest.json",
        "stats/train_class_stats.csv",
        "stats/val_class_stats.csv",
    ]:
        p = ds / rel
        assert p.exists(), f"missing dataset file/dir: {p}"
        print("OK:", p)

    n_train = len(list((ds / "images/train").glob("*.jpg")))
    n_val = len(list((ds / "images/val").glob("*.jpg")))
    assert n_train >= 1000, f"too few train images: {n_train}"
    assert n_val >= 200, f"too few val images: {n_val}"

    df = pd.read_csv(csv_path)
    print(df)

    assert len(df) >= 3, "expected baseline, proposed no-nms, proposed nms rows"

    for col in ["precision", "recall", "map50", "map50_95"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            assert ((s >= 0.0) & (s <= 1.0)).all(), f"{col} out of range"

    for col in ["preprocess_ms", "inference_ms", "postprocess_ms"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            assert (s >= 0).all(), f"{col} contains negative value"

    proposed = df[df["model"].astype(str).str.contains("proposed", case=False, na=False)]
    assert {"no-nms", "nms"}.issubset(set(proposed["postprocess"].dropna().astype(str))), "missing proposed no-nms/nms rows"

    print("Week 9 output sanity check passed")


if __name__ == "__main__":
    main()
