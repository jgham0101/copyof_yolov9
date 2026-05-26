from pathlib import Path
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    path = Path(args.csv)
    assert path.exists(), f"missing csv: {path}"
    df = pd.read_csv(path)
    print(df)

    required_cols = [
        "model", "postprocess", "precision", "recall", "map50", "map50_95",
        "preprocess_ms", "inference_ms", "postprocess_ms"
    ]
    for col in required_cols:
        assert col in df.columns, f"missing column: {col}"

    assert len(df) >= 2, "expected at least proposed no-nms and nms rows"

    for col in ["precision", "recall", "map50", "map50_95"]:
        valid = df[col].dropna()
        assert ((valid >= 0.0) & (valid <= 1.0)).all(), f"{col} out of range"

    for col in ["preprocess_ms", "inference_ms", "postprocess_ms"]:
        valid = df[col].dropna()
        assert (valid >= 0.0).all(), f"{col} contains negative values"

    proposed = df[df["model"] == "proposed_v10dual"]
    assert {"no-nms", "nms"}.issubset(set(proposed["postprocess"])), "missing proposed no-nms/nms comparison"

    print("Week 5 output sanity check passed")


if __name__ == "__main__":
    main()
