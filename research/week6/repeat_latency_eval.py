from __future__ import annotations

from pathlib import Path
import argparse
import json
import subprocess
import sys
import pandas as pd


def run_cmd(cmd: list[str], log_path: Path):
    print("$", " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, text=True, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(log_path.read_text(encoding="utf-8", errors="ignore")[-3000:])
        raise RuntimeError(f"command failed: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/coco128.yaml")
    parser.add_argument("--img", type=int, default=320)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for repeat in range(args.repeats):
        for postprocess in ["no-nms", "nms"]:
            name = f"week6_repeat_{postprocess.replace('-', '_')}_r{repeat}"
            log_path = out_dir / f"{name}.log"

            cmd = [
                sys.executable,
                "val_e2e.py",
                "--data",
                args.data,
                "--img",
                str(args.img),
                "--batch",
                str(args.batch),
                "--weights",
                args.weights,
                "--device",
                args.device,
                "--name",
                name,
                "--postprocess",
                postprocess,
            ]

            run_cmd(cmd, log_path)

            metrics_path = Path("runs/val") / name / "metrics.json"
            if not metrics_path.exists():
                raise FileNotFoundError(f"missing metrics: {metrics_path}")

            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            data["repeat"] = repeat
            data["postprocess"] = postprocess
            data["metrics_path"] = str(metrics_path)
            data["log_path"] = str(log_path)
            rows.append(data)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "week6_repeat_eval_results.csv"
    json_path = out_dir / "week6_repeat_eval_results.json"

    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(df)
    print(f"saved: {csv_path}")
    print(f"saved: {json_path}")


if __name__ == "__main__":
    main()
