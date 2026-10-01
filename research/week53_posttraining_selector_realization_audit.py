
import argparse, importlib.util, json, os
from collections import Counter
from pathlib import Path
import numpy as np
import torch, yaml
from models.experimental import attempt_load
from utils.dataloaders import create_dataloader
from utils.metrics import box_iou
from utils.loss_tal_week51_predspace_multirival import ComputeLossWeek28V2 as CaptureLoss

ROOT=Path(__file__).resolve().parents[1]
W49=ROOT/'research/week49_selector_coverage_one_negative_diagnostic.py'
spec=importlib.util.spec_from_file_location('w49',W49); w49=importlib.util.module_from_spec(spec); spec.loader.exec_module(w49)

def selector_replay(loss_fn,topk=300,iou_thr=.70,rivals_cap=3):
    logits=loss_fn._week51_bce.logits; cap=loss_fn._week51_assigner_last; out=cap['output']
    target_bboxes,target_scores,fg_mask=out[1],out[2],out[3].bool()
    pd_bboxes=loss_fn._find_pd_bboxes(cap['inputs'],target_bboxes.shape)
    boxes=pd_bboxes.detach(); prob=logits.detach().sigmoid(); fg=fg_mask.detach(); ts=target_scores.detach()
    result=[]
    for bi in range(logits.shape[0]):
        anchor_score,anchor_cls=prob[bi].max(-1); k=min(topk,logits.shape[1])
        top_score,top_idx=torch.topk(anchor_score,k=k,largest=True,sorted=True)
        top_cls=anchor_cls[top_idx]; top_boxes=boxes[bi,top_idx]; top_fg=fg[bi,top_idx]; fg_cls=ts[bi].argmax(-1)
        iou=box_iou(top_boxes.float(),top_boxes.float()); claimed=torch.zeros(k,dtype=torch.bool,device=logits.device)
        selected=set(); eligible=set(); groups=[]
        for ri in range(k):
            if bool(claimed[ri].item()): continue
            cls=int(top_cls[ri].item()); lower=torch.arange(k,device=logits.device)>ri
            cand=torch.where(lower & (~claimed) & (top_cls==cls) & (iou[ri]>=iou_thr))[0]
            if cand.numel()==0: continue
            cand_anchor=top_idx[cand]; cand_fg=fg[bi,cand_anchor]; elig=cand[~cand_fg]
            if elig.numel()==0: continue
            members=torch.cat([torch.tensor([ri],device=logits.device),cand]); member_anchor=top_idx[members]
            samepos=fg[bi,member_anchor] & (fg_cls[member_anchor]==cls)
            chosen=elig[:rivals_cap]; chosen_anchor=top_idx[chosen]
            for a in top_idx[elig]: eligible.add((int(a.item()),cls))
            for a in chosen_anchor: selected.add((int(a.item()),cls))
            residual=elig[rivals_cap:]
            sel_scores=[float(x) for x in prob[bi,chosen_anchor,cls].cpu().tolist()]
            res_scores=[float(x) for x in prob[bi,top_idx[residual],cls].cpu().tolist()] if residual.numel() else []
            groups.append({'rep_score':float(top_score[ri].item()),'eligible_count':int(elig.numel()),'selected_count':int(chosen.numel()),
                           'residual_count':int(residual.numel()),'selected_scores':sel_scores,'residual_scores':res_scores,
                           'zero_sameclass_positive':not bool(samepos.any().item()),'rep_unmatched':not bool(top_fg[ri].item())})
            claimed[chosen]=True
        result.append({'selected_keys':selected,'eligible_keys':eligible,'groups':groups})
    return result

def evaluate(label,raw_weight,proxy_weight,val,annotation,out_dir,hyp):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True); device=torch.device('cuda:0')
    proxy=attempt_load(str(proxy_weight),device=device,inplace=True,fuse=False).float().eval()
    raw=attempt_load(str(raw_weight),device=device,inplace=True,fuse=False).float(); raw.hyp=dict(yaml.safe_load(Path(hyp).read_text())); raw.train()
    for m in raw.modules():
        if isinstance(m,torch.nn.modules.batchnorm._BatchNorm): m.eval()
    if hasattr(raw.model[-1],'week28_dual_training'): raw.model[-1].week28_dual_training=True
    os.environ['YOLO_WEEK51_PREDSPACE_IOU']='0.70'; os.environ['YOLO_WEEK51_PREDSPACE_TOPK']='300'; os.environ['YOLO_WEEK51_PREDSPACE_RIVALS']='3'; os.environ['YOLO_WEEK51_PREDSPACE_LAMBDA']='0.0'
    loss=CaptureLoss(raw); stride=int(proxy.stride.max().item())
    loader=create_dataloader(str(val),640,8,stride,False,pad=.5,rect=True,workers=2,prefix=f'week53-{label}: ')[0]
    methods=['B_nonms_k300','SEL3_EXACT','SELALL_DIRECT','C70_DIAGNOSTIC']
    writers={m:w49.JsonArrayWriter(out_dir/f'{m}.json') for m in methods}; cmap=w49.coco80_to_coco91_class()
    stats=Counter(); reps=[]; sels=[]; margins=[]; residual_max=[]; residual_counts=[]; elig_counts=[]; sel_counts=[]; zero=0; unmatched=0; groups=0; align=0.0
    try:
        for im,targets,paths,shapes in loader:
            im=im.to(device).float()/255.; targets=targets.to(device)
            with torch.inference_mode():
                raw_out=raw(im); _=loss(raw_out,targets,img=im,epoch=0); decoded=w49.extract_proxy_decoded(proxy(im))
            diff=(loss._week51_bce.logits.detach().sigmoid()-decoded[:,4:,:].permute(0,2,1).contiguous()).abs(); align=max(align,float(diff.max()))
            if align>5e-4: raise RuntimeError(f'raw/proxy alignment failed {align}')
            replay=selector_replay(loss); indexed=w49.indexed_no_nms(decoded,conf=.001,max_det=300,nc=80)
            for bi in range(len(paths)):
                B=indexed[bi]['pred']; anchors=indexed[bi]['anchor']; classes=indexed[bi]['cls']; sel=replay[bi]['selected_keys']; elig=replay[bi]['eligible_keys']
                keep_sel=torch.tensor([(int(anchors[k]),int(classes[k])) not in sel for k in range(B.shape[0])],dtype=torch.bool,device=B.device)
                keep_all=torch.tensor([(int(anchors[k]),int(classes[k])) not in elig for k in range(B.shape[0])],dtype=torch.bool,device=B.device)
                variants={'B_nonms_k300':B,'SEL3_EXACT':B[keep_sel],'SELALL_DIRECT':B[keep_all],'C70_DIAGNOSTIC':w49.class_aware_c70(B,iou=.70)}
                for name,pred in variants.items():
                    w49.write_preds(writers[name],pred,paths[bi],shapes[bi],im.shape[2:],cmap)
                    stats[f'dup_{name}']+=len(w49.duplicate_pairs(pred,iou=.70)); stats[f'pred_{name}']+=int(pred.shape[0])
                stats['images']+=1
                for g in replay[bi]['groups']:
                    groups+=1; reps.append(g['rep_score']); elig_counts.append(g['eligible_count']); sel_counts.append(g['selected_count']); residual_counts.append(g['residual_count'])
                    zero+=int(g['zero_sameclass_positive']); unmatched+=int(g['rep_unmatched'])
                    for s in g['selected_scores']: sels.append(s); margins.append(g['rep_score']-s)
                    if g['residual_scores']: residual_max.append(max(g['residual_scores']))
    finally:
        for w in writers.values(): w.close()
    method_out={}
    for m in methods:
        method_out[m]={'coco_api':w49.coco_eval(annotation,out_dir/f'{m}.json'),
                       'predictions_per_image':stats[f'pred_{m}']/stats['images'],
                       'sameclass_duplicate_pairs_iou70_per_image':stats[f'dup_{m}']/stats['images']}
    mean=lambda x:float(np.mean(x)) if x else 0.0; med=lambda x:float(np.median(x)) if x else 0.0
    realization={'groups_total':groups,'groups_per_image':groups/stats['images'],'representative_score_mean':mean(reps),'representative_score_median':med(reps),
                 'selected_rival_score_mean':mean(sels),'selected_rival_score_median':med(sels),'rep_minus_selected_margin_mean':mean(margins),'rep_minus_selected_margin_median':med(margins),
                 'eligible_rivals_per_group_mean':mean(elig_counts),'selected_rivals_per_group_mean':mean(sel_counts),'residual_rivals_after_cap_per_group_mean':mean(residual_counts),
                 'groups_with_residual_after_cap_fraction':float(np.mean(np.array(residual_counts)>0)) if residual_counts else 0.0,
                 'residual_max_score_mean':mean(residual_max),'residual_max_score_median':med(residual_max),
                 'zero_sameclass_positive_group_fraction':zero/groups if groups else 0.0,'unmatched_representative_group_fraction':unmatched/groups if groups else 0.0}
    result={'label':label,'images':stats['images'],'anchor_alignment_max_abs':align,'selector_realization':realization,'methods':method_out}
    (out_dir/'week53_selector_realization.json').write_text(json.dumps(result,indent=2)); del raw,proxy,loss; torch.cuda.empty_cache(); return result

def main():
    ap=argparse.ArgumentParser()
    for x in ['label','raw-weight','proxy-weight','val','annotation','out-dir','hyp']: ap.add_argument('--'+x,required=True)
    a=ap.parse_args(); r=evaluate(a.label,Path(a.raw_weight),Path(a.proxy_weight),Path(a.val),Path(a.annotation),Path(a.out_dir),Path(a.hyp)); print(json.dumps(r,indent=2))
if __name__=='__main__': main()
