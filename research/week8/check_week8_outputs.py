
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out-dir', required=True); args = ap.parse_args()
    out = Path(args.out_dir)
    req = ['week5_week7_week8_combined_comparison.csv','final_accuracy_table.csv','final_latency_table.csv','final_proposed_postprocess_comparison.csv','final_postprocess_speedup_summary.csv','final_smallscale_tables.md','final_table_summary.json']
    for name in req:
        p = out / name
        assert p.exists(), f'missing: {p}'
        print('OK:', p)
    df = pd.read_csv(out / 'week5_week7_week8_combined_comparison.csv')
    assert len(df) >= 3
    for col in ['precision','recall','map50','map50_95']:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors='coerce').dropna()
            assert ((s >= 0) & (s <= 1)).all(), f'{col} out of range'
    for col in ['preprocess_ms','inference_ms','postprocess_ms','total_ms']:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors='coerce').dropna()
            assert (s >= 0).all(), f'{col} negative'
    assert 'week8_e10' in set(df.get('experiment', pd.Series()).dropna().astype(str)), 'week8_e10 missing'
    prop = df[df['model'].astype(str).str.contains('proposed', case=False, na=False)]
    assert {'no-nms','nms'}.issubset(set(prop['postprocess'].dropna().astype(str)))
    print('Week 8 output sanity check passed')

if __name__ == '__main__':
    main()
