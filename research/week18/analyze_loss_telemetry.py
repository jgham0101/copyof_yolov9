
import argparse
import json
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    p = Path(args.jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    csv_path = out_dir / "loss_telemetry_detail.csv"
    df.to_csv(csv_path, index=False)

    summary = {}
    if len(df):
        numeric_cols = [c for c in df.columns if c not in ["step"]]
        summary_rows = []
        for c in numeric_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                summary_rows.append({
                    "metric": c,
                    "count": int(s.notna().sum()),
                    "mean": float(s.mean()),
                    "median": float(s.median()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "first": float(s.dropna().iloc[0]),
                    "last": float(s.dropna().iloc[-1]),
                })
        sdf = pd.DataFrame(summary_rows)
        sdf.to_csv(out_dir / "loss_telemetry_summary.csv", index=False)

        # branch ratio table
        ratio = pd.DataFrame()
        ratio["step"] = df["step"]
        for a, b, name in [
            ("fg_mask_sum_o2o", "fg_mask_sum_o2m", "fg_o2o_over_o2m"),
            ("target_scores_sum_o2o", "target_scores_sum_o2m", "target_score_o2o_over_o2m"),
            ("loss_cls_o2o_raw", "loss_cls_o2m_raw", "cls_o2o_over_o2m"),
            ("loss_box_o2o_raw", "loss_box_o2m_raw", "box_o2o_over_o2m"),
            ("loss_dfl_o2o_raw", "loss_dfl_o2m_raw", "dfl_o2o_over_o2m"),
        ]:
            if a in df.columns and b in df.columns:
                aa = pd.to_numeric(df[a], errors="coerce")
                bb = pd.to_numeric(df[b], errors="coerce")
                ratio[name] = aa / bb.replace(0, pd.NA)
        ratio.to_csv(out_dir / "loss_telemetry_ratios.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "loss_telemetry_summary.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "loss_telemetry_ratios.csv", index=False)

    print("rows:", len(df))
    if len(df):
        print(df.head())
        print(pd.read_csv(out_dir / "loss_telemetry_summary.csv"))

if __name__ == "__main__":
    main()
