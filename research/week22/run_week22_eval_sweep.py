
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_cmd(cmd, cwd):
    print("$", " ".join(map(str, cmd)))

    r = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
    )

    print(r.stdout)

    if r.stderr:
        print(r.stderr)

    if r.returncode != 0:
        raise RuntimeError(" ".join(map(str, cmd)))


def latest_metrics(repo, name):
    cands = sorted(
        (repo / "runs/val").glob(name + "*/metrics.json")
    )

    if not cands:
        raise FileNotFoundError(name)

    return cands[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--conf-list", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    args = ap.parse_args()

    repo = Path(args.repo)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    confs = [
        float(x)
        for x in args.conf_list.split(",")
        if x.strip()
    ]

    rows = []

    for pp in ["nms", "no-nms"]:

        for conf in confs:

            ctag = str(conf).replace(".", "p")

            name = (
                f"week22_{args.tag}_"
                f"{pp.replace('-', '_')}_conf_{ctag}"
            )

            cmd = [
                sys.executable,
                "val_e2e.py",
                "--data", args.data,
                "--img", str(args.imgsz),
                "--batch", str(args.batch),
                "--weights", args.weights,
                "--device", args.device,
                "--name", name,
                "--postprocess", pp,
                "--conf-thres", str(conf),
                "--iou-thres", str(args.iou),
                "--max-det", str(args.max_det),
            ]

            try:
                run_cmd(cmd, repo)
                mp = latest_metrics(repo, name)
                data = json.loads(mp.read_text(encoding="utf-8"))

                row = dict(data)
                row.update({
                    "tag": args.tag,
                    "postprocess": pp,
                    "conf_thres": conf,
                    "max_det": args.max_det,
                    "metrics_path": str(mp),
                    "error": "",
                })

                rows.append(row)

            except Exception as e:
                rows.append({
                    "tag": args.tag,
                    "postprocess": pp,
                    "conf_thres": conf,
                    "max_det": args.max_det,
                    "error": repr(e),
                })

    df = pd.DataFrame(rows)

    out_csv = (
        out_dir /
        f"{args.tag}_e2e_sweep.csv"
    )

    df.to_csv(out_csv, index=False)

    print(df)
    print("saved:", out_csv)


if __name__ == "__main__":
    main()
