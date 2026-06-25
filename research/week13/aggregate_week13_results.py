
import argparse
import json
import re
from pathlib import Path
import pandas as pd

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

def clean(text):
    return ANSI_RE.sub("", text).replace("\r", "\n")

def numeric_tokens(line):
    out = []
    for tok in line.split():
        try:
            out.append(float(tok))
        except Exception:
            pass
    return out

def parse_val_log(path, label, model, postprocess, conf):
    row = {
        "label": label,
        "model": model,
        "postprocess": postprocess,
        "conf_thres": conf,
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
    if not path.exists():
        row["error"] = "missing log"
        return row

    text = clean(path.read_text(encoding="utf-8", errors="ignore"))
    candidates = []
    for line in text.splitlines():
        if re.search(r"\ball\b", line):
            nums = numeric_tokens(line)
            if len(nums) >= 6:
                candidates.append(nums)

    if candidates:
        nums = candidates[-1]
        row["images"] = int(nums[-6])
        row["instances"] = int(nums[-5])
        row["precision"] = nums[-4]
        row["recall"] = nums[-3]
        row["map50"] = nums[-2]
        row["map50_95"] = nums[-1]

    speed = list(re.finditer(
        r"Speed:\s*([0-9.]+)ms pre-process,\s*([0-9.]+)ms inference,\s*([0-9.]+)ms NMS",
        text,
    ))
    if speed:
        m = speed[-1]
        row["preprocess_ms"] = float(m.group(1))
        row["inference_ms"] = float(m.group(2))
        row["postprocess_ms"] = float(m.group(3))
    return row

def parse_metrics_json(path, label, model, conf):
    if not path.exists():
        return {"label": label, "model": model, "conf_thres": conf, "source": str(path), "error": "missing json"}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "label": label,
        "model": model,
        "postprocess": data.get("postprocess"),
        "conf_thres": conf,
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

def latest(root, pattern):
    candidates = sorted(root.glob(pattern))
    return candidates[-1] if candidates else None

def summarize_results_csv(path, label):
    if not path.exists():
        return {"label": label, "source": str(path), "error": "missing results.csv"}

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    out = {"label": label, "source": str(path), "rows": len(df)}

    numeric = df.select_dtypes(include="number")
    cols = [
        c for c in numeric.columns
        if "loss" in c.lower() or "map" in c.lower() or "precision" in c.lower() or "recall" in c.lower()
    ]

    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s):
            out[f"{c}_first"] = float(s.iloc[0])
            out[f"{c}_last"] = float(s.iloc[-1])
            out[f"{c}_min"] = float(s.min())
            out[f"{c}_max"] = float(s.max())
            out[f"{c}_delta_last_minus_first"] = float(s.iloc[-1] - s.iloc[0])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--drive", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--baseline-name", required=True)
    ap.add_argument("--proposed-name", required=True)
    args = ap.parse_args()

    repo = Path(args.repo)
    drive = Path(args.drive)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_root = repo / "runs/train"
    baseline_run = latest(train_root, args.baseline_name + "*")
    proposed_run = latest(train_root, args.proposed_name + "*")

    assert baseline_run is not None, "baseline train run not found"
    assert proposed_run is not None, "proposed train run not found"

    loss_rows = [
        summarize_results_csv(baseline_run / "results.csv", "baseline_train"),
        summarize_results_csv(proposed_run / "results.csv", "proposed_train"),
    ]
    pd.DataFrame(loss_rows).to_csv(out_dir / "loss_curve_summary.csv", index=False)

    rows = []
    for conf in [0.001, 0.1]:
        tag = str(conf).replace(".", "p")
        rows.append(parse_val_log(drive / "logs" / f"baseline_conf_{tag}.log", "baseline_native_nms", "baseline_yolov9_s", "nms", conf))
        rows.append(parse_val_log(drive / "logs" / f"proposed_native_conf_{tag}.log", "proposed_native_nms", "proposed_v10dual", "native-nms", conf))
        rows.append(parse_metrics_json(drive / "metrics_json" / f"proposed_one2one_nms_conf_{tag}.json", "proposed_one2one_nms", "proposed_v10dual", conf))
        rows.append(parse_metrics_json(drive / "metrics_json" / f"proposed_one2one_no_nms_conf_{tag}.json", "proposed_one2one_no_nms", "proposed_v10dual", conf))

    val = pd.DataFrame(rows)
    val.to_csv(out_dir / "mini_overfit_validation_summary.csv", index=False)

    evidence = {
        "baseline_run": str(baseline_run),
        "proposed_run": str(proposed_run),
        "baseline_results": str(baseline_run / "results.csv"),
        "proposed_results": str(proposed_run / "results.csv"),
        "num_validation_rows": len(val),
    }
    (out_dir / "week13_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    print("Loss summary:")
    print(pd.DataFrame(loss_rows))
    print("Validation summary:")
    print(val)
    print("saved:", out_dir)

if __name__ == "__main__":
    main()
