
from __future__ import annotations

from pathlib import Path
import argparse
import json
import pandas as pd


def get_row(df, label):
    row = df[df["label"] == label]
    if len(row) == 0:
        return None
    return row.iloc[-1]


def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--inspect-json", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    inspect_data = json.loads(Path(args.inspect_json).read_text(encoding="utf-8"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = get_row(df, "baseline_nms_reference")
    native = get_row(df, "proposed_native_val_dual")
    one2one_nms = get_row(df, "proposed_one2one_nms")
    one2one_no = get_row(df, "proposed_one2one_no_nms")

    rows = []

    def add_compare(name, a, b):
        if a is None or b is None:
            rows.append({"comparison": name, "status": "missing row"})
            return
        item = {"comparison": name, "status": "ok"}
        for key in ["precision", "recall", "map50", "map50_95", "postprocess_ms", "total_ms"]:
            av = safe_float(a.get(key))
            bv = safe_float(b.get(key))
            item[f"{key}_a"] = av
            item[f"{key}_b"] = bv
            item[f"{key}_delta_b_minus_a"] = None if av is None or bv is None else bv - av
            if av is not None and av != 0 and bv is not None:
                item[f"{key}_ratio_b_over_a"] = bv / av
            else:
                item[f"{key}_ratio_b_over_a"] = None
        rows.append(item)

    add_compare("baseline_nms -> proposed_native_val_dual", baseline, native)
    add_compare("baseline_nms -> proposed_one2one_nms", baseline, one2one_nms)
    add_compare("proposed_one2one_nms -> proposed_one2one_no_nms", one2one_nms, one2one_no)

    comp = pd.DataFrame(rows)
    comp.to_csv(out_dir / "week11_branch_delta_comparison.csv", index=False)

    notes = []
    notes.append("# Week 11 Branch Diagnosis Summary")
    notes.append("")
    notes.append("## Metrics")
    notes.append("")
    notes.append(df.to_markdown(index=False))
    notes.append("")
    notes.append("## Delta comparison")
    notes.append("")
    notes.append(comp.to_markdown(index=False))
    notes.append("")

    b_map = safe_float(baseline["map50"]) if baseline is not None else None
    n_map = safe_float(native["map50"]) if native is not None else None
    o_map = safe_float(one2one_nms["map50"]) if one2one_nms is not None else None
    no_map = safe_float(one2one_no["map50"]) if one2one_no is not None else None

    notes.append("## Initial diagnosis")
    notes.append("")

    if b_map is not None and o_map is not None and no_map is not None:
        drop_struct = b_map - o_map
        drop_nms_remove = o_map - no_map
        notes.append(f"- Baseline NMS mAP50: {b_map:.6f}")
        notes.append(f"- Proposed one2one NMS mAP50: {o_map:.6f}")
        notes.append(f"- Proposed one2one no-NMS mAP50: {no_map:.6f}")
        notes.append(f"- Drop from baseline to proposed one2one NMS: {drop_struct:.6f}")
        notes.append(f"- Drop from proposed one2one NMS to no-NMS: {drop_nms_remove:.6f}")
        if abs(drop_struct) > abs(drop_nms_remove) * 3:
            notes.append("- Primary drop is much larger before removing NMS. This suggests the main issue is likely proposed branch/head/loss quality, not NMS removal alone.")
        else:
            notes.append("- NMS removal contributes substantially to the drop. no-NMS postprocess should be investigated first.")

    if n_map is not None and o_map is not None:
        diff_native_one2one = abs(n_map - o_map)
        notes.append("")
        notes.append(f"- Proposed native val_dual mAP50: {n_map:.6f}")
        notes.append(f"- Proposed one2one NMS mAP50: {o_map:.6f}")
        notes.append(f"- Absolute difference: {diff_native_one2one:.6f}")
        if diff_native_one2one < 0.001:
            notes.append("- Native val_dual result is very close to one2one NMS. The model's default inference path may be using a similar branch/output, or both branches are similarly weak.")
        else:
            notes.append("- Native val_dual differs from one2one NMS. This may indicate that default/native output and explicit one2one output are not identical.")

    notes.append("")
    notes.append("## Output structure evidence")
    notes.append("")
    modules = inspect_data.get("detect_modules", [])
    notes.append(f"- Detect-like modules found: {modules}")
    notes.append(f"- Number of output structure entries: {len(inspect_data.get('output_structure', []))}")
    notes.append("")
    notes.append("See `branch_output_structure.json` and `v10dual_forward_source.txt` for raw evidence.")

    (out_dir / "week11_diagnosis_summary.md").write_text("\\n".join(notes), encoding="utf-8")

    evidence = {
        "baseline_map50": b_map,
        "proposed_native_map50": n_map,
        "proposed_one2one_nms_map50": o_map,
        "proposed_one2one_no_nms_map50": no_map,
        "detect_modules": modules,
    }
    (out_dir / "week11_diagnosis_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    print("\\n".join(notes))
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
