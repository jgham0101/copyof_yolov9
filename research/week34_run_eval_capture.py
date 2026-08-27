from __future__ import annotations
import argparse, json, re, shutil, subprocess, sys
from pathlib import Path

INTERNAL_RE = re.compile(
    r"\ball\s+\d+\s+\d+\s+"
    r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)"
)
COCO_AP95_RE = re.compile(
    r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50:0\.95\s*"
    r"\|\s*area=\s*all\s*\|\s*maxDets=\s*100\s*\]\s*=\s*([0-9.]+)"
)
COCO_AP50_RE = re.compile(
    r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50\s*"
    r"\|\s*area=\s*all\s*\|\s*maxDets=\s*100\s*\]\s*=\s*([0-9.]+)"
)
COCO_AP75_RE = re.compile(
    r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.75\s*"
    r"\|\s*area=\s*all\s*\|\s*maxDets=\s*100\s*\]\s*=\s*([0-9.]+)"
)

def one(rx, text, name):
    hits = rx.findall(text)
    if len(hits) != 1:
        raise RuntimeError(f"Expected one {name}, got {hits}")
    return float(hits[0])

def parse(text):
    internal = None
    for line in text.splitlines():
        m = INTERNAL_RE.search(line)
        if m:
            internal = {
                "precision": float(m.group(1)),
                "recall": float(m.group(2)),
                "map50": float(m.group(3)),
                "map50_95": float(m.group(4)),
            }
    if internal is None:
        raise RuntimeError("Internal metric row missing.")
    return {
        "internal": internal,
        "coco_api_map50_95": one(COCO_AP95_RE, text, "COCO AP50-95"),
        "coco_api_map50": one(COCO_AP50_RE, text, "COCO AP50"),
        "coco_api_ap75": one(COCO_AP75_RE, text, "COCO AP75"),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--script", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, required=True)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--prediction-json-out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    save_dir = Path(a.project) / a.name
    if save_dir.exists():
        shutil.rmtree(save_dir)

    cmd = [
        sys.executable, a.script,
        "--data", "data/coco.yaml",
        "--weights", a.weights,
        "--batch-size", str(a.batch),
        "--img", "640",
        "--conf-thres", str(a.conf),
        "--iou-thres", "0.7",
        "--max-det", str(a.max_det),
        "--device", "0",
        "--workers", str(a.workers),
        "--save-json",
        "--project", a.project,
        "--name", a.name,
        "--exist-ok",
    ]

    p = subprocess.run(
        cmd, cwd=a.repo, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )

    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    Path(a.log).write_text(p.stdout, encoding="utf-8")
    print(p.stdout)

    if p.returncode != 0:
        raise RuntimeError(f"Evaluator failed rc={p.returncode}. See {a.log}")
    if "Traceback" in p.stdout:
        raise RuntimeError(f"Unexpected traceback. See {a.log}")

    result = parse(p.stdout)

    pred_files = sorted(save_dir.glob("*_predictions.json"))
    if len(pred_files) != 1:
        raise RuntimeError(
            f"Expected one predictions JSON under {save_dir}, got {pred_files}"
        )

    pred_out = Path(a.prediction_json_out)
    pred_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pred_files[0], pred_out)

    result.update({
        "script": a.script,
        "weights": a.weights,
        "conf_thres": a.conf,
        "max_det": a.max_det,
        "iou_thres": 0.7,
        "batch": a.batch,
        "workers": a.workers,
        "prediction_json": str(pred_out),
    })

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
