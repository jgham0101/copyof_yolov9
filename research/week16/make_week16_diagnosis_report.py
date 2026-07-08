
import argparse, json
from pathlib import Path
import pandas as pd


def read_csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def summarize_train(run_dir):
    p = Path(run_dir) / 'results.csv'
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    out = {'run': str(run_dir), 'rows': int(len(df))}
    for c in df.columns:
        if c.strip() in ['train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95']:
            s = pd.to_numeric(df[c], errors='coerce')
            out[c.strip() + '_first'] = float(s.iloc[0]) if len(s) else None
            out[c.strip() + '_last'] = float(s.iloc[-1]) if len(s) else None
            out[c.strip() + '_max'] = float(s.max()) if len(s) else None
            out[c.strip() + '_min'] = float(s.min()) if len(s) else None
    return out


def best_by_tag(df):
    out = []
    if len(df) == 0 or 'tag' not in df.columns:
        return pd.DataFrame()
    tmp = df.copy()
    for c in ['precision', 'recall', 'map50', 'map50_95', 'postprocess_ms']:
        if c in tmp.columns:
            tmp[c] = pd.to_numeric(tmp[c], errors='coerce')
    for tag in sorted(tmp['tag'].dropna().unique()):
        sub = tmp[tmp['tag'] == tag].dropna(subset=['map50'])
        if len(sub):
            out.append(sub.sort_values('map50', ascending=False).iloc[0].to_dict())
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drive', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    drive = Path(args.drive)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_dir = drive / 'sweeps'
    sweep_files = sorted(sweep_dir.glob('*_val_e2e_sweep.csv'))
    sweeps = []
    for f in sweep_files:
        df = pd.read_csv(f)
        df['sweep_file'] = f.name
        sweeps.append(df)
    all_sweeps = pd.concat(sweeps, ignore_index=True, sort=False) if sweeps else pd.DataFrame()
    all_sweeps.to_csv(out_dir / 'week16_all_e2e_sweeps.csv', index=False)

    best = best_by_tag(all_sweeps)
    best.to_csv(out_dir / 'week16_best_by_tag.csv', index=False)

    native_files = sorted((drive / 'native_val').glob('*_native_val_metrics.csv'))
    native = pd.concat([pd.read_csv(f) for f in native_files], ignore_index=True, sort=False) if native_files else pd.DataFrame()
    native.to_csv(out_dir / 'week16_native_val_summary.csv', index=False)

    selected = json.loads((drive / 'selected_week16_weights.json').read_text(encoding='utf-8')) if (drive / 'selected_week16_weights.json').exists() else {}
    train_rows = []
    for k, v in selected.items():
        if k.endswith('_run'):
            train_rows.append({'tag': k.replace('_run', ''), **summarize_train(v)})
    train_df = pd.DataFrame(train_rows)
    train_df.to_csv(out_dir / 'week16_train_summary.csv', index=False)

    lines = []
    lines.append('# Week 16 COCO128 Improvement Validation Summary')
    lines.append('')
    lines.append('## Purpose')
    lines.append('')
    lines.append('Week 16 moves from mini-overfit diagnosis to COCO128 validation. It tests whether Week 15 improvement strategies generalize beyond the 32-image mini-overfit setting.')
    lines.append('')
    lines.append('## Native validation summary')
    lines.append('')
    lines.append(native.to_markdown(index=False) if len(native) else 'No native validation table.')
    lines.append('')
    lines.append('## E2E best-by-tag')
    lines.append('')
    lines.append(best.to_markdown(index=False) if len(best) else 'No E2E best table.')
    lines.append('')
    lines.append('## All E2E sweeps')
    lines.append('')
    cols = [c for c in ['tag', 'postprocess', 'conf_thres', 'precision', 'recall', 'map50', 'map50_95', 'preprocess_ms', 'inference_ms', 'postprocess_ms', 'error'] if c in all_sweeps.columns]
    lines.append(all_sweeps[cols].to_markdown(index=False) if len(all_sweeps) else 'No E2E sweep table.')
    lines.append('')
    lines.append('## Train summary')
    lines.append('')
    lines.append(train_df.to_markdown(index=False) if len(train_df) else 'No train summary table.')
    lines.append('')
    lines.append('## Interpretation guide')
    lines.append('')
    lines.append('- If `proposed_from_baseline` beats `proposed_scratch`, Week 14/15 feature-initialization diagnosis generalizes to COCO128.')
    lines.append('- If `proposed_one2one_copy` further beats `proposed_from_baseline`, one-to-one branch initialization remains useful beyond mini-overfit.')
    lines.append('- If no-NMS is close to NMS after improvement, NMS removal itself is not the dominant cause on COCO128 either.')
    lines.append('- If improvement disappears on COCO128, the mini-overfit result may be overfit-specific and training recipe/loss/assigner should be revisited.')
    lines.append('')

    (out_dir / 'week16_diagnosis_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    evidence = {
        'sweep_files': [str(f) for f in sweep_files],
        'native_files': [str(f) for f in native_files],
        'num_e2e_rows': int(len(all_sweeps)),
        'num_native_rows': int(len(native)),
        'selected': selected,
    }
    (out_dir / 'week16_evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
