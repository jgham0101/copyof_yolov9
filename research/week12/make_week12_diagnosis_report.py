
from __future__ import annotations

from pathlib import Path
import argparse
import json
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction-dir", required=True)
    ap.add_argument("--sweep-csv", required=True)
    ap.add_argument("--week11-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    pred_dir = Path(args.prediction_dir)
    sweep_csv = Path(args.sweep_csv)
    week11_csv = Path(args.week11_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_summary = pd.read_csv(pred_dir / "prediction_aggregate_summary.csv")
    pred_count = pd.read_csv(pred_dir / "prediction_count_confidence_summary.csv")
    sweep = pd.read_csv(sweep_csv)
    week11 = pd.read_csv(week11_csv)

    lines = []
    lines.append("# Week 12 Prediction / Confidence Diagnosis Summary")
    lines.append("")
    lines.append("## Week 11 reference")
    lines.append("")
    lines.append(week11.to_markdown(index=False))
    lines.append("")
    lines.append("## Prediction aggregate summary")
    lines.append("")
    lines.append(pred_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## Threshold sweep")
    lines.append("")
    lines.append(sweep.to_markdown(index=False))
    lines.append("")
    lines.append("## Diagnostic reading guide")
    lines.append("")
    lines.append("- If proposed modes have much larger `mean_num_pred` than baseline, false positive over-generation is likely.")
    lines.append("- If proposed confidence values are very low or threshold-sensitive, confidence calibration is likely unstable.")
    lines.append("- If mAP improves at higher confidence thresholds, the main issue is likely false positives / score calibration.")
    lines.append("- If mAP stays low across thresholds, bbox/class prediction quality or decode/branch selection should be inspected.")
    lines.append("- If prediction overlays show shifted or wrongly scaled boxes, decode/scale coordinate logic should be checked.")
    lines.append("")

    lines.append("## Automatic notes")
    lines.append("")

    try:
        baseline = pred_summary[pred_summary["label"] == "baseline_native_nms"].iloc[-1]
        proposed_no = pred_summary[pred_summary["label"] == "proposed_one2one_no_nms"].iloc[-1]
        proposed_nms = pred_summary[pred_summary["label"] == "proposed_one2one_nms"].iloc[-1]
        lines.append(f"- Baseline mean predictions per image: {baseline['mean_num_pred']:.3f}")
        lines.append(f"- Proposed one2one NMS mean predictions per image: {proposed_nms['mean_num_pred']:.3f}")
        lines.append(f"- Proposed one2one no-NMS mean predictions per image: {proposed_no['mean_num_pred']:.3f}")
        if proposed_no["mean_num_pred"] > baseline["mean_num_pred"] * 2:
            lines.append("- Proposed no-NMS produces far more predictions than baseline. False positive over-generation is likely.")
        elif proposed_no["mean_num_pred"] < baseline["mean_num_pred"] * 0.5:
            lines.append("- Proposed no-NMS produces far fewer predictions than baseline. Low-confidence suppression or missing detections may be likely.")
        else:
            lines.append("- Proposed no-NMS prediction count is in the same broad range as baseline.")
    except Exception as e:
        lines.append(f"- Automatic prediction-count interpretation skipped: {repr(e)}")

    if "map50" in sweep.columns and "postprocess" in sweep.columns:
        for mode in ["no-nms", "nms"]:
            sub = sweep[sweep["postprocess"] == mode].copy()
            if len(sub):
                sub["map50"] = pd.to_numeric(sub["map50"], errors="coerce")
                best = sub.sort_values("map50", ascending=False).iloc[0]
                lines.append(f"- Best threshold for {mode} by mAP50: conf={best['conf_thres']}, mAP50={best['map50']}")

    (out_dir / "week12_diagnosis_summary.md").write_text("\\n".join(lines), encoding="utf-8")
    pred_summary.to_csv(out_dir / "prediction_aggregate_summary.csv", index=False)
    pred_count.to_csv(out_dir / "prediction_count_confidence_summary.csv", index=False)
    sweep.to_csv(out_dir / "threshold_sweep_results.csv", index=False)

    evidence = {
        "prediction_dir": str(pred_dir),
        "sweep_csv": str(sweep_csv),
        "week11_csv": str(week11_csv),
        "num_prediction_rows": len(pred_count),
        "num_sweep_rows": len(sweep),
    }
    (out_dir / "week12_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    print("\\n".join(lines))
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
