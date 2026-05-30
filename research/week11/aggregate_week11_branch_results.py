
from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
import pandas as pd


def parse_float_list(line: str):
    vals = []
    for token in line.strip().split():
        try:
            vals.append(float(token))
        except ValueError:
            pass
    return vals


def parse_val_log(path: Path, model: str, postprocess: str, label: str) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    metrics = {
        "label": label,
        "model": model,
        "postprocess": postprocess,
        "images": None,
        "instances": None,
        "precision": None,
        "recall": None,
        "map50": None,
        "map50_95": None,
        "preprocess_ms": None,
        "inference_ms": None,
        "postprocess_ms": None,
        "source": str(path),
    }

    for line in reversed(text.splitlines()):
        if re.search(r"\\ball\\b", line):
            nums = parse_float_list(line)
            if len(nums) >= 6:
                metrics["images"] = int(nums[-6])
                metrics["instances"] = int(nums[-5])
                metrics["precision"] = nums[-4]
                metrics["recall"] = nums[-3]
                metrics["map50"] = nums[-2]
                metrics["map50_95"] = nums[-1]
                break
            elif len(nums) >= 4:
                metrics["precision"] = nums[-4]
                metrics["recall"] = nums[-3]
                metrics["map50"] = nums[-2]
                metrics["map50_95"] = nums[-1]
                break

    speed_match = re.search(
        r"Speed:\\s*([0-9.]+)ms pre-process,\\s*([0-9.]+)ms inference,\\s*([0-9.]+)ms NMS",
        text,
    )
    if speed_match:
        metrics["preprocess_ms"] = float(speed_match.group(1))
        metrics["inference_ms"] = float(speed_match.group(2))
        metrics["postprocess_ms"] = float(speed_match.group(3))

    return metrics


def parse_metrics_json(path: Path, model: str, label: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "label": label,
        "model": model,
        "postprocess": data.get("postprocess"),
        "images": data.get("images"),
        "instances": data.get("instances"),
        "precision": data.get("precision"),
        "recall": data.get("recall"),
        "map50": data.get("map50"),
        "map50_95": data.get("map50_95"),
        "preprocess_ms": data.get("preprocess_ms"),
        "inference_ms": data.get("inference_ms"),
        "postprocess_ms": data.get("postprocess_ms"),
        "source": str(path),
    }


def latest(root: Path, pattern: str):
    ms = sorted(root.rglob(pattern))
    return ms[-1] if ms else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    drive = Path(args.drive)
    repo = Path(args.repo)

    rows = []

    baseline_log = drive / "logs" / "week11_baseline_reference_val_dual.log"
    proposed_native_log = drive / "logs" / "week11_proposed_native_val_dual.log"

    rows.append(parse_val_log(baseline_log, "baseline_yolov9_s", "nms", "baseline_nms_reference"))
    rows.append(parse_val_log(proposed_native_log, "proposed_v10dual", "native-val_dual-nms", "proposed_native_val_dual"))

    one2one_no = latest(repo, "week11_proposed_one2one_no_nms_eval*/metrics.json")
    one2one_nms = latest(repo, "week11_proposed_one2one_nms_eval*/metrics.json")

    if one2one_nms:
        rows.append(parse_metrics_json(one2one_nms, "proposed_v10dual", "proposed_one2one_nms"))
    if one2one_no:
        rows.append(parse_metrics_json(one2one_no, "proposed_v10dual", "proposed_one2one_no_nms"))

    df = pd.DataFrame(rows)

    numeric = ["precision", "recall", "map50", "map50_95", "preprocess_ms", "inference_ms", "postprocess_ms"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if {"preprocess_ms", "inference_ms", "postprocess_ms"}.issubset(df.columns):
        df["total_ms"] = df[["preprocess_ms", "inference_ms", "postprocess_ms"]].sum(axis=1, skipna=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(df)
    print("saved:", out)


if __name__ == "__main__":
    main()
