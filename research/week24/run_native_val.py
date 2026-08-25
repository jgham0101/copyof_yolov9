
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

# IMPORTANT:
# COCO prints:
#
# Average Precision (AP) @[ IoU=0.50:0.95 | area= all |
# maxDets=100 ] = 0.468
#
# We must capture the number AFTER "] =",
# not the 100 in maxDets=100.

COCO_AP95_RE = re.compile(
    r"Average Precision\s+\(AP\)\s+@\[\s*"
    r"IoU=0\.50:0\.95"
    r".*?"
    r"\]\s*=\s*([0-9.]+)"
)

COCO_AP50_RE = re.compile(
    r"Average Precision\s+\(AP\)\s+@\[\s*"
    r"IoU=0\.50\s+\|"
    r".*?"
    r"\]\s*=\s*([0-9.]+)"
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

    out = rows[-1] if rows else {}

    # Parse COCO API lines one line at a time.
    # This prevents matching unrelated '=' tokens.
    for line in text.splitlines():

        m95 = COCO_AP95_RE.search(line)
        if m95:
            out["coco_ap50_95"] = float(m95.group(1))

        m50 = COCO_AP50_RE.search(line)
        if m50:
            out["coco_ap50"] = float(m50.group(1))

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

    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)

    ap.add_argument("--device", default="0")

    ap.add_argument(
        "--batch-candidates",
        default="32,16,8",
    )

    ap.add_argument(
        "--save-json",
        action="store_true",
    )

    args = ap.parse_args()

    batches = [
        int(x)
        for x in args.batch_candidates.split(",")
        if x.strip()
    ]

    final = None

    for batch in batches:

        cmd = [
            sys.executable,
            args.script,

            "--data",
            args.data,

            "--img",
            str(args.imgsz),

            "--batch",
            str(batch),

            "--conf",
            str(args.conf),

            "--iou",
            str(args.iou),

            "--device",
            args.device,

            "--weights",
            args.weights,

            "--project",
            args.project,

            "--name",
            args.name,

            "--exist-ok",
        ]

        if args.save_json:
            cmd.append("--save-json")

        print(
            "$",
            " ".join(cmd),
            flush=True,
        )

        r = subprocess.run(
            cmd,
            cwd=args.repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        text = r.stdout or ""

        print(text)

        Path(args.log).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        Path(args.log).write_text(
            text,
            encoding="utf-8",
        )

        if r.returncode == 0:

            final = {
                "batch": batch,
                "returncode": 0,
                **parse(text),
            }

            break

        if "out of memory" not in text.lower():

            raise RuntimeError(
                f"Validation failed. "
                f"See {args.log}"
            )

        print(
            f"OOM at batch={batch}; "
            "retry smaller batch."
        )

    if final is None:
        raise RuntimeError(
            "All batch candidates failed."
        )

    # Sanity protection against the exact parser bug
    if "coco_ap50_95" in final:
        assert 0.0 <= final["coco_ap50_95"] <= 1.0, (
            "Invalid COCO AP50-95 parsed: "
            f"{final['coco_ap50_95']}"
        )

    if "coco_ap50" in final:
        assert 0.0 <= final["coco_ap50"] <= 1.0, (
            "Invalid COCO AP50 parsed: "
            f"{final['coco_ap50']}"
        )

    Path(args.json_out).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(args.json_out).write_text(
        json.dumps(
            final,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            final,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
