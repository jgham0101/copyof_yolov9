
import argparse
import json
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-dir", required=True)
    ap.add_argument("--gradient-baseline", required=True)
    ap.add_argument("--gradient-proposed", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    summary_dir = Path(args.summary_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loss = pd.read_csv(summary_dir / "loss_curve_summary.csv")
    val = pd.read_csv(summary_dir / "mini_overfit_validation_summary.csv")
    gb = json.loads(Path(args.gradient_baseline).read_text(encoding="utf-8"))
    gp = json.loads(Path(args.gradient_proposed).read_text(encoding="utf-8"))

    lines = []
    lines.append("# Week 13 Overfit / Loss / Gradient Diagnosis Summary")
    lines.append("")
    lines.append("## Loss curve summary")
    lines.append("")
    lines.append(loss.to_markdown(index=False))
    lines.append("")
    lines.append("## Mini-overfit validation summary")
    lines.append("")
    lines.append(val.to_markdown(index=False))
    lines.append("")
    lines.append("## Gradient probe summary")
    lines.append("")
    lines.append(f"- Baseline params with gradient: {gb.get('params_with_grad')} / {gb.get('total_params')}")
    lines.append(f"- Proposed params with gradient: {gp.get('params_with_grad')} / {gp.get('total_params')}")
    lines.append("")
    lines.append("### Baseline detect modules")
    lines.append("")
    lines.append(pd.DataFrame(gb.get("detect_modules", [])).to_markdown(index=False))
    lines.append("")
    lines.append("### Proposed detect modules")
    lines.append("")
    lines.append(pd.DataFrame(gp.get("detect_modules", [])).to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("- If baseline overfits but proposed does not, proposed head/loss/branch learning is likely the issue.")
    lines.append("- If proposed also overfits, the architecture can learn and the issue may be training recipe/confidence calibration.")
    lines.append("- If detect gradients are zero, branch/loss connection should be inspected.")
    lines.append("- If train loss decreases but mini-overfit mAP remains poor, decode/postprocess/class score calibration should be inspected.")

    (out_dir / "week13_diagnosis_summary.md").write_text("\\n".join(lines), encoding="utf-8")
    (out_dir / "week13_evidence.json").write_text(json.dumps({
        "loss_rows": len(loss),
        "validation_rows": len(val),
        "baseline_gradient": args.gradient_baseline,
        "proposed_gradient": args.gradient_proposed,
    }, indent=2), encoding="utf-8")

    print("\\n".join(lines))

if __name__ == "__main__":
    main()
