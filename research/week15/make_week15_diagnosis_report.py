
import argparse, json
from pathlib import Path
import pandas as pd

def read_csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def best_by_tag(df):
    rows=[]
    if len(df)==0 or 'tag' not in df.columns: return pd.DataFrame()
    tmp=df.copy()
    for c in ['precision','recall','map50','map50_95','postprocess_ms']:
        if c in tmp.columns: tmp[c]=pd.to_numeric(tmp[c], errors='coerce')
    for tag in sorted(tmp['tag'].dropna().unique()):
        sub=tmp[tmp['tag']==tag].dropna(subset=['map50'])
        if len(sub): rows.append(sub.sort_values('map50', ascending=False).iloc[0].to_dict())
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--week14-sweeps', required=True); ap.add_argument('--week15-sweep-dir', required=True); ap.add_argument('--copy-report', required=True); ap.add_argument('--out-dir', required=True)
    args=ap.parse_args(); out_dir=Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    week14=read_csv(args.week14_sweeps)
    tabs=[]
    for f in sorted(Path(args.week15_sweep_dir).glob('*_val_e2e_sweep.csv')):
        df=pd.read_csv(f); df['sweep_file']=f.name; tabs.append(df)
    week15=pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame()
    combined=pd.concat([week14, week15], ignore_index=True, sort=False) if len(week14) else week15
    combined.to_csv(out_dir/'week15_all_sweeps_with_week14_reference.csv', index=False)
    best=best_by_tag(combined); best.to_csv(out_dir/'week15_best_by_tag.csv', index=False)
    copy_report=json.loads(Path(args.copy_report).read_text(encoding='utf-8')) if Path(args.copy_report).exists() else {}
    cols=[c for c in ['tag','postprocess','conf_thres','precision','recall','map50','map50_95','postprocess_ms','error'] if c in combined.columns]
    lines=['# Week 15 Improvement / Causal Ablation Summary','', '## One2one copy initialization report','```json',json.dumps(copy_report, indent=2),'```','', '## Best result by experiment tag', best.to_markdown(index=False) if len(best) else 'No best table.', '', '## All sweep results', combined[cols].to_markdown(index=False) if len(combined) else 'No sweep table.', '', '## Interpretation guide', '- If `week15_one2one_copy_init` improves over `proposed_from_baseline`, one-to-one branch initialization is a major cause.', '- If `week15_detach_off_from_baseline` improves over `proposed_from_baseline`, detach-limited feature learning is a major cause.', '- If no-NMS catches up after improvements, no-NMS itself is not the dominant cause; one-to-one branch quality is.']
    (out_dir/'week15_diagnosis_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    (out_dir/'week15_evidence.json').write_text(json.dumps({'num_week15_rows': int(len(week15)), 'num_combined_rows': int(len(combined)), 'copy_report': copy_report}, indent=2), encoding='utf-8')
    print('\n'.join(lines))
if __name__=='__main__': main()
