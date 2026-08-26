import argparse,json,sys
from pathlib import Path
import pandas as pd,torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from models.yolo import V10O2MDetect,V10DualDormantDetect
from utils.general import non_max_suppression

def load(p):
    try:return torch.load(p,map_location='cpu',weights_only=False)
    except TypeError:return torch.load(p,map_location='cpu')
def unwrap(c): return (c.get('ema') or c.get('model')) if isinstance(c,dict) else c
def cmp(name,a,b):
    if a.shape!=b.shape:return {'name':name,'shape_equal':False,'max_abs_diff':float('inf'),'mean_abs_diff':float('inf'),'allclose':False}
    d=(a.float()-b.float()).abs(); return {'name':name,'shape_equal':True,'max_abs_diff':float(d.max()) if d.numel() else 0.,'mean_abs_diff':float(d.mean()) if d.numel() else 0.,'allclose':bool(torch.allclose(a.float(),b.float(),rtol=1e-5,atol=1e-6))}
def fhooks(m,s):
    hs=[]
    for i in [15,18,21]:
        def h(_m,_x,o,i=i):s[i]=o.detach().clone()
        hs.append(m.model[i].register_forward_hook(h))
    return hs

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--reference'); ap.add_argument('--dormant'); ap.add_argument('--out-dir'); a=ap.parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ref=unwrap(load(a.reference)).float().to(dev).eval(); dor=unwrap(load(a.dormant)).float().to(dev).eval()
    assert isinstance(ref.model[-1],V10O2MDetect) and isinstance(dor.model[-1],V10DualDormantDetect)
    calls={'one2one_cv2':0,'one2one_cv3':0}
    h2=dor.model[-1].one2one_cv2.register_forward_hook(lambda m,i,o:calls.__setitem__('one2one_cv2',calls['one2one_cv2']+1))
    h3=dor.model[-1].one2one_cv3.register_forward_hook(lambda m,i,o:calls.__setitem__('one2one_cv3',calls['one2one_cv3']+1))
    rf,df={},{}; hs=fhooks(ref,rf)+fhooks(dor,df)
    torch.manual_seed(2026); x=torch.rand(1,3,640,640,device=dev)
    with torch.no_grad(): ro=ref(x); do=dor(x)
    for h in hs+[h2,h3]:h.remove()
    rdec,rraw=ro[0],ro[1]; ddec,draw=do[0],do[1]
    fdf=pd.DataFrame([cmp(f'layer_{i}',rf[i],df[i]) for i in [15,18,21]]); fdf.to_csv(out/'week26_feature_equivalence.csv',index=False)
    rdf=pd.DataFrame([cmp(f'P{i+3}_raw',r,d) for i,(r,d) in enumerate(zip(rraw,draw))]); rdf.to_csv(out/'week26_o2m_raw_equivalence.csv',index=False)
    dec=cmp('decoded',rdec,ddec); (out/'week26_decode_equivalence.json').write_text(json.dumps(dec,indent=2))
    rn=non_max_suppression(rdec.clone(),0.001,0.7,multi_label=True,max_det=300); dn=non_max_suppression(ddec.clone(),0.001,0.7,multi_label=True,max_det=300)
    rows=[]
    for i,(r,d) in enumerate(zip(rn,dn)):
        z={'batch_index':i,'reference_count':int(r.shape[0]),'dormant_count':int(d.shape[0]),'max_abs_diff':float('inf'),'allclose':False}
        if r.shape==d.shape:
            q=(r.float()-d.float()).abs(); z['max_abs_diff']=float(q.max()) if q.numel() else 0.; z['allclose']=bool(torch.allclose(r.float(),d.float(),rtol=1e-5,atol=1e-6))
        rows.append(z)
    ndf=pd.DataFrame(rows); ndf.to_csv(out/'week26_nms_equivalence.csv',index=False)
    crep={'one2one_cv2_forward_calls':calls['one2one_cv2'],'one2one_cv3_forward_calls':calls['one2one_cv3'],'total_o2o_forward_calls':sum(calls.values()),'pass_zero_calls':sum(calls.values())==0}; (out/'week26_dormant_call_report.json').write_text(json.dumps(crep,indent=2))
    gate={'dormant_zero_call_pass':crep['pass_zero_calls'],'feature_pass':bool(fdf.allclose.all()),'raw_o2m_pass':bool(rdf.allclose.all()),'decode_pass':bool(dec['allclose']),'nms_pass':bool(ndf.allclose.all()),'feature_max_abs_diff':float(fdf.max_abs_diff.max()),'raw_max_abs_diff':float(rdf.max_abs_diff.max()),'decode_max_abs_diff':float(dec['max_abs_diff']),'nms_max_abs_diff':float(ndf.max_abs_diff.max())}; (out/'week26_tensor_gate.json').write_text(json.dumps(gate,indent=2)); print(json.dumps(crep,indent=2)); print(json.dumps(gate,indent=2))
if __name__=='__main__':main()
