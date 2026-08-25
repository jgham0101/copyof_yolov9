from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ALL_RE = re.compile(
    r"^\s*all\s+(\d+)\s+(\d+)\s+"
    r"([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+"
    r"([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*$"
)

SPEED_RE = re.compile(
    r"Speed:\s*([0-9.]+)ms pre-process,\s*"
    r"([0-9.]+)ms inference,\s*"
    r"([0-9.]+)ms NMS per image"
)

def parse(text):
    rows = []
    for line in text.splitlines():
        m = ALL_RE.match(line)
        if m:
            rows.append({
                "images": int(m.group(1)),
                "instances": int(m.group(2)),
                "precision": float(m.group(3)),
                "recall": float(m.group(4)),
                "map50": float(m.group(5)),
                "map50_95": float(m.group(6)),
            })

    if not rows:
        raise RuntimeError("Could not parse final 'all' metrics row.")

    out = rows[-1]
    m = SPEED_RE.search(text)
    if m:
        out["preprocess_ms"] = float(m.group(1))
        out["inference_ms"] = float(m.group(2))
        out["postprocess_ms"] = float(m.group(3))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--script", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--batches", default="16,8")
    a = ap.parse_args()

    final = None

    for batch in [int(x) for x in a.batches.split(",") if x.strip()]:
        cmd = [
            sys.executable, a.script,
            "--data", a.data,
            "--img", "640",
            "--batch", str(batch),
            "--conf", "0.001",
            "--iou", "0.7",
            "--device", "0",
            "--weights", a.weights,
            "--project", a.project,
            "--name", a.name,
            "--exist-ok",
        ]

        print("$", " ".join(cmd))
        r = subprocess.run(
            cmd, cwd=a.repo, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )

        text = r.stdout or ""
        print(text)

        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        Path(a.log).write_text(text, encoding="utf-8")

        if r.returncode == 0:
            final = {"batch": batch, **parse(text)}
            break

        if "out of memory" not in text.lower():
            raise RuntimeError(f"Validation failed. See {a.log}")

    if final is None:
        raise RuntimeError("All batch candidates failed")

    Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json_out).write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )
    print(json.dumps(final, indent=2))

if __name__ == "__main__":
    main()
