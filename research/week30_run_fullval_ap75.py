from __future__ import annotations
import argparse, json, re, subprocess, sys
from pathlib import Path

INTERNAL_RE = re.compile(
    r"all\s+\d+\s+\d+\s+"
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

def parse(text):
    internal = None
    for line in text.splitlines():
        if "all" in line:
            m = INTERNAL_RE.search(line)
            if m:
                internal = {
                    "precision": float(m.group(1)),
                    "recall": float(m.group(2)),
                    "map50": float(m.group(3)),
                    "map50_95": float(m.group(4)),
                }

    if internal is None:
        raise RuntimeError("Internal metric line not found.")

    def one(regex, name):
        matches = regex.findall(text)
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one {name} area=all result, got {matches}"
            )
        return float(matches[0])

    return {
        "internal": internal,
        "coco_api_map50_95": one(COCO_AP95_RE, "AP50-95"),
        "coco_api_map50": one(COCO_AP50_RE, "AP50"),
        "coco_api_ap75": one(COCO_AP75_RE, "AP75"),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    cmd = [
        sys.executable, "val.py",
        "--data", "data/coco.yaml",
        "--weights", a.weights,
        "--batch-size", str(a.batch),
        "--img", "640",
        "--conf-thres", "0.001",
        "--iou-thres", "0.7",
        "--device", "0",
        "--save-json",
        "--project", a.project,
        "--name", a.name,
        "--exist-ok",
    ]

    p = subprocess.run(
        cmd,
        cwd=a.repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    Path(a.log).write_text(p.stdout, encoding="utf-8")
    print(p.stdout)

    if p.returncode != 0:
        raise RuntimeError(f"val.py failed rc={p.returncode}")

    result = parse(p.stdout)
    result["command"] = cmd

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
