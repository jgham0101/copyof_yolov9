
from __future__ import annotations

from pathlib import Path
import argparse
import json
import subprocess
import sys

import pandas as pd


def run_cmd(cmd: list[str], cwd: Path):
    print("$", " ".join(map(str, cmd)))
    r = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    print(r.stdout)
    if r.stderr:
        print(r.stderr)
    if r.returncode != 0:
        raise RuntimeError(" ".join(map(str, cmd)))
    return r


def find_metrics(repo: Path, name: str):
    candidates = sorted((repo / "runs/val").glob(f"{name}*/metrics.json"))
    if not candidates:
        raise FileNotFoundError(f"metrics not found: {name}")
    return candidates[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--conf-list", required=True, help="comma-separated values")
    args = ap.parse_args()

    repo = Path(args.repo)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    confs = [float(x) for x in args.conf_list.split(",") if x.strip()]
    rows = []

    for mode in ["no-nms", "nms"]:
        for conf in confs:
            tag = f"week12_sweep_{mode.replace('-', '_')}_conf_{str(conf).replace('.', 'p')}"
            cmd = [
                sys.executable,
                "val_e2e.py",
                "--data",
                args.data,
                "--img",
                str(args.imgsz),
                "--batch",
                str(args.batch),
                "--weights",
                args.weights,
                "--device",
                args.device,
                "--name",
                tag,
                "--postprocess",
                mode,
                "--conf-thres",
                str(conf),
                "--iou-thres",
                str(args.iou),
                "--max-det",
                str(args.max_det),
            ]

            try:
                run_cmd(cmd, cwd=repo)
                metrics_path = find_metrics(repo, tag)
                data = json.loads(metrics_path.read_text(encoding="utf-8"))
                row = {
                    "postprocess": mode,
                    "conf_thres": conf,
                    "metrics_path": str(metrics_path),
                    **data,
                }
                rows.append(row)

                dst = out_dir / f"{tag}_metrics.json"
                dst.write_text(json.dumps(data, indent=2), encoding="utf-8")

            except Exception as e:
                rows.append({
                    "postprocess": mode,
                    "conf_thres": conf,
                    "error": repr(e),
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "threshold_sweep_results.csv", index=False)
    print(df)
    print("saved:", out_dir / "threshold_sweep_results.csv")


if __name__ == "__main__":
    main()
