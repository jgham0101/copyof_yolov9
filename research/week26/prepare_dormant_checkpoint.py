import argparse,copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from models.yolo import Model,V10O2MDetect,V10DualDormantDetect

def load(p):
    try:return torch.load(p,map_location='cpu',weights_only=False)
    except TypeError:return torch.load(p,map_location='cpu')
def unwrap(c): return (c.get('ema') or c.get('model')) if isinstance(c,dict) else c

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source'); ap.add_argument('--cfg'); ap.add_argument('--out'); ap.add_argument('--report'); a=ap.parse_args()
    src=unwrap(load(a.source)).float().eval(); assert isinstance(src.model[-1],V10O2MDetect)
    tgt=Model(a.cfg,ch=3,nc=src.nc,anchors=None).float(); assert isinstance(tgt.model[-1],V10DualDormantDetect)
    ss,ts=src.state_dict(),tgt.state_dict(); si=len(src.model)-1; ti=len(tgt.model)-1; sp=f'model.{si}.'; tp=f'model.{ti}.'
    mapped={}; o2o={}
    for k,v in ts.items():
        if '.one2one_cv2.' in k or '.one2one_cv3.' in k: continue
        if k not in ss or ss[k].shape!=v.shape: raise RuntimeError(f'common map failure {k}')
        mapped[k]=ss[k].clone()
    for sstem,tstem in [('cv2.','one2one_cv2.'),('cv3.','one2one_cv3.')]:
        sb=sp+sstem; tb=tp+tstem
        for sk in [k for k in ss if k.startswith(sb)]:
            tk=tb+sk[len(sb):]
            if tk not in ts or ss[sk].shape!=ts[tk].shape: raise RuntimeError(f'o2o map failure {sk}->{tk}')
            mapped[tk]=ss[sk].clone(); o2o[tk]=sk
    missing=[k for k in ts if k not in mapped]
    if missing: raise RuntimeError('unmapped: '+str(missing[:20]))
    tgt.load_state_dict(mapped,strict=True); tgt.names=copy.deepcopy(src.names); tgt.nc=src.nc
    loaded=tgt.state_dict(); common_max=0.; o2o_max=0.
    for k,v in mapped.items():
        ref=ss[o2o[k]] if k in o2o else ss[k]
        d=(loaded[k].float()-ref.float()).abs(); md=float(d.max()) if d.numel() else 0.
        if k in o2o:o2o_max=max(o2o_max,md)
        else:common_max=max(common_max,md)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    torch.save({'epoch':-1,'best_fitness':None,'model':copy.deepcopy(tgt).half(),'ema':None,'optimizer':None,'updates':None},a.out)
    rep={'target_state_items':len(ts),'mapped_state_items':len(mapped),'dormant_o2o_state_items':len(o2o),'unmapped_target_items':len(missing),'common_postload_max_abs_diff':common_max,'o2o_vs_o2m_init_max_abs_diff':o2o_max,'dormant_o2o_mapping':o2o}
    Path(a.report).write_text(json.dumps(rep,indent=2)); print(json.dumps({k:v for k,v in rep.items() if k!='dormant_o2o_mapping'},indent=2))
if __name__=='__main__':main()
