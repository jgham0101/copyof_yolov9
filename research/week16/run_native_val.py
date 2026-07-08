
import argparse, json, re, subprocess, sys
from pathlib import Path
import pandas as pd


def parse_all_line(text):
    # Typical line: all 128 929 0.1 0.2 0.3 0.15
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith('all'):
            continue
        parts = s.split()
        nums = []
        for p in parts[1:]:
            try:
                nums.append(float(p))
            except Exception:
                pass
        if len(nums) >= 6:
            rows.append(nums)
    if not rows:
        return {}
    nums = rows[-1]
    return {
        'images': int(nums[0]),
        'instances': int(nums[1]),
        'precision': nums[2],
        'recall': nums[3],
        'map50': nums[4],
        'map50_95': nums[5],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--device', default='0')
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--iou', type=float, default=0.7)
    args = ap.parse_args()

    repo = Path(args.repo)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f'week16_{args.tag}_native_val'
    cmd = [
        sys.executable, 'val_dual.py',
        '--data', args.data,
        '--img', str(args.imgsz),
        '--batch', str(args.batch),
        '--weights', args.weights,
        '--device', args.device,
        '--name', name,
        '--conf-thres', str(args.conf),
        '--iou-thres', str(args.iou),
    ]
    print('$', ' '.join(cmd))
    r = subprocess.run(cmd, cwd=repo, text=True, capture_output=True)
    log = r.stdout + '\n' + r.stderr
    print(log)
    (out_dir / f'{tag if False else name}.log').write_text(log, encoding='utf-8')
    if r.returncode != 0:
        raise RuntimeError('native val failed')
    metrics = {'tag': args.tag, 'postprocess': 'native-val', 'conf_thres': args.conf, 'weights': args.weights, **parse_all_line(log)}
    (out_dir / f'{args.tag}_native_val_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    pd.DataFrame([metrics]).to_csv(out_dir / f'{args.tag}_native_val_metrics.csv', index=False)
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
