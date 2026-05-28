from __future__ import annotations
from pathlib import Path
import argparse, json
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv', required=True); ap.add_argument('--out-dir', required=True); args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df=pd.read_csv(args.csv)
    for c in ['precision','recall','map50','map50_95','preprocess_ms','inference_ms','postprocess_ms']:
        if c in df.columns: df[c]=pd.to_numeric(df[c], errors='coerce')
    if {'preprocess_ms','inference_ms','postprocess_ms'}.issubset(df.columns):
        df['total_ms']=df[['preprocess_ms','inference_ms','postprocess_ms']].sum(axis=1, skipna=False)
    acc=df[[c for c in ['model','postprocess','precision','recall','map50','map50_95'] if c in df.columns]]
    lat=df[[c for c in ['model','postprocess','preprocess_ms','inference_ms','postprocess_ms','total_ms'] if c in df.columns]]
    prop=df[df['model'].astype(str).str.contains('proposed',case=False,na=False)].copy()
    acc.to_csv(out/'voc2007_accuracy_table.csv', index=False); lat.to_csv(out/'voc2007_latency_table.csv', index=False); prop.to_csv(out/'voc2007_proposed_postprocess_comparison.csv', index=False)
    no=prop[prop['postprocess'].astype(str).str.contains('no',case=False,na=False)]; nms=prop[prop['postprocess'].astype(str).str.lower()=='nms']
    row={'dataset':'VOC2007'}
    if len(no) and len(nms):
        no_ms=pd.to_numeric(no.iloc[-1]['postprocess_ms'], errors='coerce'); nms_ms=pd.to_numeric(nms.iloc[-1]['postprocess_ms'], errors='coerce')
        row['no_nms_postprocess_ms']=float(no_ms) if pd.notna(no_ms) else None; row['nms_postprocess_ms']=float(nms_ms) if pd.notna(nms_ms) else None
        row['speedup_nms_over_no_nms']=(row['nms_postprocess_ms']/row['no_nms_postprocess_ms']) if row.get('no_nms_postprocess_ms') and row.get('nms_postprocess_ms') is not None else None
    speed=pd.DataFrame([row]); speed.to_csv(out/'voc2007_postprocess_speedup_summary.csv', index=False)
    md = '# Week 10 VOC2007 Benchmark Tables\n\n## Accuracy Table\n' + acc.to_markdown(index=False) + '\n\n## Latency Table\n' + lat.to_markdown(index=False) + '\n\n## Proposed Postprocess Comparison\n' + prop.to_markdown(index=False) + '\n\n## Postprocess Speedup Summary\n' + speed.to_markdown(index=False)
    (out/'voc2007_tables.md').write_text(md, encoding='utf-8')
    (out/'voc2007_table_summary.json').write_text(json.dumps({'rows':len(df),'outputs':['voc2007_accuracy_table.csv','voc2007_latency_table.csv','voc2007_proposed_postprocess_comparison.csv','voc2007_postprocess_speedup_summary.csv','voc2007_tables.md']}, indent=2), encoding='utf-8')
    print(df); print(speed); print('saved:', out)
if __name__ == '__main__': main()
