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


def parse_val_dual_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    metrics = {
        "model": "baseline_yolov9_s",
        "postprocess": "nms",
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
        if re.search(r"\ball\b", line) and len(parse_float_list(line)) >= 4:
            nums = parse_float_list(line)
            metrics["precision"], metrics["recall"], metrics["map50"], metrics["map50_95"] = nums[-4:]
            break

    speed_match = re.search(
        r"Speed:\s*([0-9.]+)ms pre-process,\s*([0-9.]+)ms inference,\s*([0-9.]+)ms NMS",
        text,
    )
    if speed_match:
        metrics["preprocess_ms"] = float(speed_match.group(1))
        metrics["inference_ms"] = float(speed_match.group(2))
        metrics["postprocess_ms"] = float(speed_match.group(3))

    return metrics


def normalize_e2e_metrics(path: Path, postprocess: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "model": "proposed_v10dual",
        "postprocess": postprocess,
        "precision": data.get("precision"),
        "recall": data.get("recall"),
        "map50": data.get("map50"),
        "map50_95": data.get("map50_95"),
        "preprocess_ms": data.get("preprocess_ms"),
        "inference_ms": data.get("inference_ms"),
        "postprocess_ms": data.get("postprocess_ms"),
        "source": str(path),
    }


def find_latest(root: Path, pattern: str):
    matches = sorted(root.rglob(pattern))
    return matches[-1] if matches else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--drive", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    repo = Path(args.repo)
    drive = Path(args.drive)
    rows = []

    rows.append(parse_val_dual_log(drive / "logs" / "week7_baseline_val_dual.log"))

    no_nms_metrics = find_latest(repo, "week7_proposed_no_nms_eval*/metrics.json")
    nms_metrics = find_latest(repo, "week7_proposed_nms_eval*/metrics.json")

    if no_nms_metrics:
        rows.append(normalize_e2e_metrics(no_nms_metrics, "no-nms"))
    if nms_metrics:
        rows.append(normalize_e2e_metrics(nms_metrics, "nms"))

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(df)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
