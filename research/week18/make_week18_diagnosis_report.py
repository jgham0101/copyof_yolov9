
import argparse
import json
from pathlib import Path
import pandas as pd

def read_csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week17-best", required=True)
    ap.add_argument("--week17-sweeps", required=True)
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--telemetry-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    week17_best = read_csv(args.week17_best)
    week17_sweeps = read_csv(args.week17_sweeps)

    probe_tables = []
    shape_tables = []
    for f in sorted(Path(args.probe_dir).glob("*_nonms_probe_aggregate.csv")):
        df = pd.read_csv(f)
        df["source_file"] = f.name
        probe_tables.append(df)
    for f in sorted(Path(args.probe_dir).glob("*_output_shapes.csv")):
        df = pd.read_csv(f)
        df["source_file"] = f.name
        shape_tables.append(df)
    probe = pd.concat(probe_tables, ignore_index=True) if probe_tables else pd.DataFrame()
    shapes = pd.concat(shape_tables, ignore_index=True) if shape_tables else pd.DataFrame()

    telemetry_detail = read_csv(Path(args.telemetry_dir) / "loss_telemetry_detail.csv")
    telemetry_summary = read_csv(Path(args.telemetry_dir) / "loss_telemetry_summary.csv")
    telemetry_ratios = read_csv(Path(args.telemetry_dir) / "loss_telemetry_ratios.csv")

    week17_best.to_csv(out_dir / "week18_week17_best_reference.csv", index=False)
    probe.to_csv(out_dir / "week18_nonms_probe_aggregate_all.csv", index=False)
    shapes.to_csv(out_dir / "week18_nonms_output_shapes.csv", index=False)
    telemetry_detail.to_csv(out_dir / "week18_loss_telemetry_detail.csv", index=False)
    telemetry_summary.to_csv(out_dir / "week18_loss_telemetry_summary.csv", index=False)
    telemetry_ratios.to_csv(out_dir / "week18_loss_telemetry_ratios.csv", index=False)

    lines = []
    lines.append("# Week 18 no-NMS Probe Repair / Loss-Assigner Telemetry Summary")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Week 18 uses the Week 17 low-augmentation model as the reference, repairs box-level no-NMS probing, and logs one2many/one2one assignment and loss telemetry.")
    lines.append("")
    lines.append("## Week 17 reference best results")
    lines.append("")
    lines.append(week17_best.to_markdown(index=False) if len(week17_best) else "No Week17 reference table.")
    lines.append("")
    lines.append("## no-NMS probe aggregate")
    lines.append("")
    lines.append(probe.to_markdown(index=False) if len(probe) else "No no-NMS probe aggregate.")
    lines.append("")
    lines.append("## loss telemetry summary")
    lines.append("")
    lines.append(telemetry_summary.to_markdown(index=False) if len(telemetry_summary) else "No telemetry summary.")
    lines.append("")
    lines.append("## loss telemetry ratio tail")
    lines.append("")
    lines.append(telemetry_ratios.tail(20).to_markdown(index=False) if len(telemetry_ratios) else "No telemetry ratio table.")
    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("- If one2one_no_nms now has valid rows with zero errors, the Week17 probe failure was a script/normalization issue.")
    lines.append("- If one2one_no_nms produces boxes but lower class-match IoU than NMS, score ranking or top-k selection is the core gap.")
    lines.append("- If fg_mask_sum_o2o is consistently far below fg_mask_sum_o2m, one2one positive assignment is weak.")
    lines.append("- If one2one cls/box/dfl telemetry is unstable or high while o2m is stable, branch-specific loss balance should be patched next.")
    lines.append("")

    (out_dir / "week18_diagnosis_summary.md").write_text("\n".join(lines), encoding="utf-8")

    evidence = {
        "num_probe_rows": int(len(probe)),
        "num_shape_rows": int(len(shapes)),
        "num_telemetry_rows": int(len(telemetry_detail)),
        "num_telemetry_summary_rows": int(len(telemetry_summary)),
    }
    (out_dir / "week18_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
