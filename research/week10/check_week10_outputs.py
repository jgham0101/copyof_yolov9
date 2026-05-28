from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv', required=True); ap.add_argument('--dataset-root', required=True); ap.add_argument('--tables-dir', required=True); args=ap.parse_args()
    csv_path, ds, tables = Path(args.csv), Path(args.dataset_root), Path(args.tables_dir)
    assert csv_path.exists(), csv_path; assert ds.exists(), ds
    for rel in ['images/train','images/val','labels/train','labels/val','voc2007_manifest.json','stats/train_class_stats.csv','stats/val_class_stats.csv']:
        p=ds/rel; assert p.exists(), f'missing {p}'; print('OK:', p)
    ntr=len(list((ds/'images/train').glob('*.jpg'))); nva=len(list((ds/'images/val').glob('*.jpg')))
    nltr=len(list((ds/'labels/train').glob('*.txt'))); nlva=len(list((ds/'labels/val').glob('*.txt')))
    print('train/val:', ntr, nltr, nva, nlva)
    assert ntr >= 5000 and nva >= 4900 and nltr == ntr and nlva == nva
    df=pd.read_csv(csv_path); print(df); assert len(df) >= 3
    for c in ['precision','recall','map50','map50_95']:
        if c in df.columns:
            s=pd.to_numeric(df[c], errors='coerce').dropna(); assert ((s>=0)&(s<=1)).all(), c
    for c in ['preprocess_ms','inference_ms','postprocess_ms']:
        if c in df.columns:
            s=pd.to_numeric(df[c], errors='coerce').dropna(); assert (s>=0).all(), c
    prop=df[df['model'].astype(str).str.contains('proposed',case=False,na=False)]
    assert {'no-nms','nms'}.issubset(set(prop['postprocess'].dropna().astype(str)))
    for name in ['voc2007_accuracy_table.csv','voc2007_latency_table.csv','voc2007_proposed_postprocess_comparison.csv','voc2007_postprocess_speedup_summary.csv','voc2007_tables.md','voc2007_table_summary.json']:
        p=tables/name; assert p.exists(), p; print('OK:', p)
    print('Week 10 output sanity check passed')
if __name__ == '__main__': main()
