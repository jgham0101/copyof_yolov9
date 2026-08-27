from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


INTERNAL_RE = re.compile(
    r"all\s+\d+\s+\d+\s+"
    r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)"
)

SPEED_RE = re.compile(
    r"Speed:\s*"
    r"([0-9.]+)ms\s*pre-process,\s*"
    r"([0-9.]+)ms\s*inference,\s*"
    r"([0-9.]+)ms\s*NMS"
)


def parse(text):
    internal = None
    speed = None

    for line in text.splitlines():

        m = INTERNAL_RE.search(line)

        if m and "all" in line:
            internal = {
                "precision": float(m.group(1)),
                "recall": float(m.group(2)),
                "map50": float(m.group(3)),
                "map50_95": float(m.group(4)),
            }

        s = SPEED_RE.search(line)

        if s:
            speed = {
                "preprocess_ms": float(s.group(1)),
                "inference_ms": float(s.group(2)),
                "postprocess_ms": float(s.group(3)),
                "total_ms": (
                    float(s.group(1))
                    + float(s.group(2))
                    + float(s.group(3))
                ),
            }

    if internal is None:
        raise RuntimeError(
            "Internal metric row not found."
        )

    return {
        "metrics": internal,
        "speed": speed,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", required=True, type=float)
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--batch", type=int, default=16)

    a = ap.parse_args()

    cmd = [
        sys.executable,
        "val.py",
        "--data",
        "data/coco.yaml",
        "--weights",
        a.weights,
        "--batch-size",
        str(a.batch),
        "--img",
        "640",
        "--conf-thres",
        str(a.conf),
        "--iou-thres",
        "0.7",
        "--device",
        "0",
        "--project",
        a.project,
        "--name",
        a.name,
        "--exist-ok",
    ]

    p = subprocess.run(
        cmd,
        cwd=a.repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    Path(a.log).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(a.log).write_text(
        p.stdout,
        encoding="utf-8",
    )

    print(p.stdout)

    if p.returncode != 0:
        raise RuntimeError(
            f"val.py failed rc={p.returncode}"
        )

    result = parse(p.stdout)

    result.update({
        "conf_thres": a.conf,
        "weights": a.weights,
        "command": cmd,
    })

    Path(a.out).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(a.out).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
