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
        if re.search(r"\\ball\\b", line) and len(parse_float_list(line)) >= 4:
            nums = parse_float_list(line)
            metrics["precision"], metrics["recall"], metrics["map50"], metrics["map50_95"] = nums[-4:]
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


def latest(root: Path, pattern: str):
    ms = sorted(root.rglob(pattern))
    return ms[-1] if ms else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--drive", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo = Path(args.repo)
    drive = Path(args.drive)

    rows = [parse_val_dual_log(drive / "logs" / "week9_baseline_val_dual.log")]

    no = latest(repo, "week9_proposed_no_nms_eval*/metrics.json")
    nm = latest(repo, "week9_proposed_nms_eval*/metrics.json")

    if no:
        rows.append(normalize_e2e_metrics(no, "no-nms"))
    if nm:
        rows.append(normalize_e2e_metrics(nm, "nms"))

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(df)
    print("saved:", out)


if __name__ == "__main__":
    main()
