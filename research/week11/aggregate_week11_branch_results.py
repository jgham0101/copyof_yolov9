from __future__ import annotations

from pathlib import Path
import argparse
import json
import re

import pandas as pd


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def clean_log_text(text: str) -> str:
    """Remove ANSI escape codes and normalize progress-bar carriage returns."""
    text = ANSI_RE.sub("", text)
    text = text.replace("\r", "\n")
    return text


def numeric_tokens(line: str) -> list[float]:
    vals = []
    for token in line.strip().split():
        try:
            vals.append(float(token))
        except ValueError:
            pass
    return vals


def parse_val_dual_log(path: Path, model: str, postprocess: str, label: str) -> dict:
    """
    Parse YOLO val_dual.py terminal log.

    Expected result line:
        all 1000 8307 0.556 0.0356 0.0293 0.0158

    Expected speed line:
        Speed: 0.1ms pre-process, 10.4ms inference, 1.9ms NMS per image ...
    """
    if not path.exists():
        raise FileNotFoundError(f"missing validation log: {path}")

    text = clean_log_text(path.read_text(encoding="utf-8", errors="ignore"))

    row = {
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

    all_candidates = []

    for line in text.splitlines():
        if re.search(r"\ball\b", line):
            nums = numeric_tokens(line)

            # Use last 6 numbers:
            # Images, Instances, P, R, mAP50, mAP50-95
            if len(nums) >= 6:
                all_candidates.append((line, nums))

    if not all_candidates:
        tail = "\n".join(text.splitlines()[-80:])
        raise RuntimeError(
            f"could not parse 'all' metric line from {path}\n"
            f"--- log tail ---\n{tail}"
        )

    all_line, nums = all_candidates[-1]
    row["images"] = int(nums[-6])
    row["instances"] = int(nums[-5])
    row["precision"] = float(nums[-4])
    row["recall"] = float(nums[-3])
    row["map50"] = float(nums[-2])
    row["map50_95"] = float(nums[-1])

    speed_matches = list(
        re.finditer(
            r"Speed:\s*([0-9.]+)ms pre-process,\s*([0-9.]+)ms inference,\s*([0-9.]+)ms NMS",
            text,
        )
    )

    if not speed_matches:
        tail = "\n".join(text.splitlines()[-80:])
        raise RuntimeError(
            f"could not parse Speed line from {path}\n"
            f"--- log tail ---\n{tail}"
        )

    speed = speed_matches[-1]
    row["preprocess_ms"] = float(speed.group(1))
    row["inference_ms"] = float(speed.group(2))
    row["postprocess_ms"] = float(speed.group(3))

    print(f"[parsed] {label}")
    print("  all line:", all_line)
    print(
        "  speed:",
        row["preprocess_ms"],
        row["inference_ms"],
        row["postprocess_ms"],
    )

    return row


def parse_metrics_json(path: Path, model: str, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing metrics json: {path}")

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


def latest(paths: list[Path]) -> Path | None:
    paths = [p for p in paths if p.exists()]
    return sorted(paths)[-1] if paths else None


def find_metrics_json(repo: Path, drive: Path, run_name: str) -> Path:
    """
    Prefer Drive runs_backup because Colab runtime may be fresh.
    Fall back to local repo/runs/val.
    """
    candidates = []

    candidates.extend((drive / "runs_backup").glob(f"{run_name}*/metrics.json"))
    candidates.extend((repo / "runs/val").glob(f"{run_name}*/metrics.json"))

    selected = latest(candidates)
    if selected is None:
        searched = [
            str(drive / "runs_backup" / f"{run_name}*/metrics.json"),
            str(repo / "runs/val" / f"{run_name}*/metrics.json"),
        ]
        raise FileNotFoundError(f"metrics.json not found for {run_name}. searched={searched}")

    print(f"[selected metrics] {run_name}: {selected}")
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    drive = Path(args.drive)
    repo = Path(args.repo)

    baseline_log = drive / "logs" / "week11_baseline_reference_val_dual.log"
    proposed_native_log = drive / "logs" / "week11_proposed_native_val_dual.log"

    one2one_nms_json = find_metrics_json(
        repo,
        drive,
        "week11_proposed_one2one_nms_eval",
    )

    one2one_no_nms_json = find_metrics_json(
        repo,
        drive,
        "week11_proposed_one2one_no_nms_eval",
    )

    rows = [
        parse_val_dual_log(
            baseline_log,
            model="baseline_yolov9_s",
            postprocess="nms",
            label="baseline_nms_reference",
        ),
        parse_val_dual_log(
            proposed_native_log,
            model="proposed_v10dual",
            postprocess="native-val_dual-nms",
            label="proposed_native_val_dual",
        ),
        parse_metrics_json(
            one2one_nms_json,
            model="proposed_v10dual",
            label="proposed_one2one_nms",
        ),
        parse_metrics_json(
            one2one_no_nms_json,
            model="proposed_v10dual",
            label="proposed_one2one_no_nms",
        ),
    ]

    df = pd.DataFrame(rows)

    numeric_cols = [
        "images",
        "instances",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["total_ms"] = df[["preprocess_ms", "inference_ms", "postprocess_ms"]].sum(
        axis=1,
        skipna=False,
    )

    metric_cols = ["precision", "recall", "map50", "map50_95"]
    if df[metric_cols].isna().any().any():
        raise RuntimeError(f"NaN remains in metric columns:\n{df}")

    latency_cols = ["preprocess_ms", "inference_ms", "postprocess_ms"]
    if df[latency_cols].isna().any().any():
        raise RuntimeError(f"NaN remains in latency columns:\n{df}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out, index=False)
    out.with_suffix(".json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )

    print(df)
    print("saved:", out)


if __name__ == "__main__":
    main()
