from __future__ import annotations
from pathlib import Path
import argparse, json, re
import pandas as pd

def nums(line):
    out=[]
    for t in line.strip().split():
        try: out.append(float(t))
        except ValueError: pass
    return out

def parse_log(path: Path):
    text = path.read_text(encoding='utf-8', errors='ignore') if path.exists() else ''
    m = {'model':'baseline_yolov9_s','postprocess':'nms','precision':None,'recall':None,'map50':None,'map50_95':None,'preprocess_ms':None,'inference_ms':None,'postprocess_ms':None,'source':str(path)}
    for line in reversed(text.splitlines()):
        if re.search(r'\ball\b', line) and len(nums(line)) >= 4:
            m['precision'], m['recall'], m['map50'], m['map50_95'] = nums(line)[-4:]
            break
    sm = re.search(r'Speed:\s*([0-9.]+)ms pre-process,\s*([0-9.]+)ms inference,\s*([0-9.]+)ms NMS', text)
    if sm:
        m['preprocess_ms'] = float(sm.group(1)); m['inference_ms'] = float(sm.group(2)); m['postprocess_ms'] = float(sm.group(3))
    return m

def norm(path: Path, postprocess: str):
    d = json.loads(path.read_text(encoding='utf-8'))
    return {'model':'proposed_v10dual','postprocess':postprocess,'precision':d.get('precision'),'recall':d.get('recall'),'map50':d.get('map50'),'map50_95':d.get('map50_95'),'preprocess_ms':d.get('preprocess_ms'),'inference_ms':d.get('inference_ms'),'postprocess_ms':d.get('postprocess_ms'),'source':str(path)}

def latest(root: Path, pattern: str):
    ms = sorted(root.rglob(pattern)); return ms[-1] if ms else None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--repo', required=True); ap.add_argument('--drive', required=True); ap.add_argument('--out', required=True); args = ap.parse_args()
    repo, drive = Path(args.repo), Path(args.drive)
    rows=[parse_log(drive/'logs'/'week10_baseline_val_dual.log')]
    no, nm = latest(repo,'week10_proposed_no_nms_eval*/metrics.json'), latest(repo,'week10_proposed_nms_eval*/metrics.json')
    if no: rows.append(norm(no,'no-nms'))
    if nm: rows.append(norm(nm,'nms'))
    df=pd.DataFrame(rows); out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out,index=False); out.with_suffix('.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(df); print('saved:', out)
if __name__ == '__main__': main()
