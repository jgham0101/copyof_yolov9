
import argparse
import json
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    args = ap.parse_args()

    root = Path(args.drive)
    required = [
        "week13_experiment_config.json",
        "week13_mini32_overfit.yaml",
        "mini_dataset_manifest.json",
        "hyp.week13-overfit.yaml",
        "gradient_probe/baseline_gradient_probe.json",
        "gradient_probe/proposed_gradient_probe.json",
        "summary/loss_curve_summary.csv",
        "summary/mini_overfit_validation_summary.csv",
        "summary/week13_evidence.json",
        "week13_diagnosis_report/week13_diagnosis_summary.md",
        "week13_diagnosis_report/week13_evidence.json",
        "runs_backup",
    ]

    for rel in required:
        p = root / rel
        assert p.exists(), f"missing: {p}"
        print("OK:", p)

    val = pd.read_csv(root / "summary/mini_overfit_validation_summary.csv")
    loss = pd.read_csv(root / "summary/loss_curve_summary.csv")
    assert len(val) >= 8, "expected at least 8 validation rows"
    assert len(loss) >= 2, "expected baseline/proposed loss rows"

    for rel in ["gradient_probe/baseline_gradient_probe.json", "gradient_probe/proposed_gradient_probe.json"]:
        data = json.loads((root / rel).read_text(encoding="utf-8"))
        assert data["params_with_grad"] > 0, f"no gradients: {rel}"
        assert len(data["detect_modules"]) > 0, f"no detect modules: {rel}"

    print("Week 13 output sanity check passed")

if __name__ == "__main__":
    main()
