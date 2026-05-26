from pathlib import Path
import argparse
import csv
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-nms', required=True)
    parser.add_argument('--nms', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    no_nms = json.loads(Path(args.no_nms).read_text())
    nms = json.loads(Path(args.nms).read_text())

    rows = []
    for d in [nms, no_nms]:
        rows.append({
            'postprocess': d['postprocess'],
            'images': d['images'],
            'instances': d['instances'],
            'precision': d['precision'],
            'recall': d['recall'],
            'map50': d['map50'],
            'map50_95': d['map50_95'],
            'preprocess_ms': d['preprocess_ms'],
            'inference_ms': d['inference_ms'],
            'postprocess_ms': d['postprocess_ms'],
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    speedup = nms['postprocess_ms'] / max(no_nms['postprocess_ms'], 1e-12)
    print('Week 4 metric comparison')
    print(f'NMS postprocess ms    : {nms["postprocess_ms"]:.6f}')
    print(f'no-NMS postprocess ms : {no_nms["postprocess_ms"]:.6f}')
    print(f'postprocess speedup   : {speedup:.4f}x')
    print(f'NMS mAP50-95          : {nms["map50_95"]:.6f}')
    print(f'no-NMS mAP50-95       : {no_nms["map50_95"]:.6f}')
    print(f'CSV saved to          : {out}')


if __name__ == '__main__':
    main()
