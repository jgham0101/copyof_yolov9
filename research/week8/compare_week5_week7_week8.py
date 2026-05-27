
from __future__ import annotations
from pathlib import Path
import argparse, json
import pandas as pd

def load_csv(path: Path, label: str):
    if not path.exists():
        print('missing', label, path); return None
    df = pd.read_csv(path); df['experiment'] = label; return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--week5', required=True); ap.add_argument('--week7', required=True); ap.add_argument('--week8', required=True); ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    frames = [x for x in [load_csv(Path(args.week5),'week5_e1'), load_csv(Path(args.week7),'week7_e3'), load_csv(Path(args.week8),'week8_e10')] if x is not None]
    if not frames: raise FileNotFoundError('No comparison CSV files found.')
    df = pd.concat(frames, ignore_index=True)
    for col in ['precision','recall','map50','map50_95','preprocess_ms','inference_ms','postprocess_ms']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    if {'preprocess_ms','inference_ms','postprocess_ms'}.issubset(df.columns):
        df['total_ms'] = df[['preprocess_ms','inference_ms','postprocess_ms']].sum(axis=1, skipna=False)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'week5_week7_week8_combined_comparison.csv', index=False)
    acc = df[[c for c in ['experiment','model','postprocess','precision','recall','map50','map50_95'] if c in df.columns]]
    lat = df[[c for c in ['experiment','model','postprocess','preprocess_ms','inference_ms','postprocess_ms','total_ms'] if c in df.columns]]
    prop = df[df['model'].astype(str).str.contains('proposed', case=False, na=False)].copy()
    acc.to_csv(out / 'final_accuracy_table.csv', index=False)
    lat.to_csv(out / 'final_latency_table.csv', index=False)
    prop.to_csv(out / 'final_proposed_postprocess_comparison.csv', index=False)
    rows = []
    for exp, g in prop.groupby('experiment'):
        no = g[g['postprocess'].astype(str).str.contains('no', case=False, na=False)]
        nms = g[g['postprocess'].astype(str).str.lower() == 'nms']
        row = {'experiment': exp}
        if len(no) and len(nms):
            no_ms = pd.to_numeric(no.iloc[-1]['postprocess_ms'], errors='coerce')
            nms_ms = pd.to_numeric(nms.iloc[-1]['postprocess_ms'], errors='coerce')
            row['no_nms_postprocess_ms'] = float(no_ms) if pd.notna(no_ms) else None
            row['nms_postprocess_ms'] = float(nms_ms) if pd.notna(nms_ms) else None
            row['speedup_nms_over_no_nms'] = (row['nms_postprocess_ms']/row['no_nms_postprocess_ms']) if row.get('no_nms_postprocess_ms') and row.get('nms_postprocess_ms') is not None else None
        rows.append(row)
    speed = pd.DataFrame(rows)
    speed.to_csv(out / 'final_postprocess_speedup_summary.csv', index=False)
    md = '# Final Small-Scale Experiment Tables\\n\\n## Accuracy Table\\n' + acc.to_markdown(index=False) + '\\n\\n## Latency Table\\n' + lat.to_markdown(index=False) + '\\n\\n## Proposed Postprocess Comparison\\n' + prop.to_markdown(index=False) + '\\n\\n## Postprocess Speedup Summary\\n' + speed.to_markdown(index=False)
    (out / 'final_smallscale_tables.md').write_text(md, encoding='utf-8')
    summary = {'rows': len(df), 'experiments': sorted(df['experiment'].dropna().astype(str).unique().tolist())}
    (out / 'final_table_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(df); print(speed); print('saved:', out)

if __name__ == '__main__':
    main()
