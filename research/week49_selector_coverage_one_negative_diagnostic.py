import argparse,json,os
from pathlib import Path
from collections import Counter
import numpy as np, torch, yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import nms
from models.experimental import attempt_load
from utils.dataloaders import create_dataloader
from utils.general import check_img_size,non_max_suppression,scale_boxes,xywh2xyxy,xyxy2xywh,coco80_to_coco91_class
from utils.metrics import box_iou
from utils.loss_tal_week48_duplicate_hn import ComputeLossWeek28V2 as CaptureLoss
from utils.week46_nonms_postprocess import _to_b84n

class JsonArrayWriter:
    def __init__(self,path): self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self.f=self.path.open('w'); self.f.write('['); self.first=True; self.count=0
    def write_obj(self,obj):
        if not self.first:self.f.write(',')
        self.f.write(json.dumps(obj,separators=(',',':'))); self.first=False; self.count+=1
    def close(self): self.f.write(']'); self.f.close()

def coco_eval(annotation,pred_json):
    gt=COCO(str(annotation)); dt=gt.loadRes(str(pred_json)); ev=COCOeval(gt,dt,'bbox'); ev.evaluate(); ev.accumulate(); ev.summarize(); s=ev.stats
    return {'ap50_95':float(s[0]),'ap50':float(s[1]),'ap75':float(s[2]),'ar1':float(s[6]),'ar10':float(s[7]),'ar100':float(s[8]),'coco_standard_maxdets':[1,10,100]}

def extract_proxy_decoded(raw,nc=80):
    if torch.is_tensor(raw): y=raw
    elif isinstance(raw,(list,tuple)) and len(raw)>=1 and torch.is_tensor(raw[0]): y=raw[0]
    else: raise RuntimeError(f'Unsupported proxy output: {type(raw)}')
    y=_to_b84n(y,nc=nc)
    if y.ndim!=3 or y.shape[1]!=4+nc: raise RuntimeError(tuple(y.shape))
    return y

def indexed_no_nms(decoded,conf=.001,max_det=300,nc=80):
    b,ch,n=decoded.shape; assert ch==4+nc
    boxes=decoded[:,:4,:].permute(0,2,1).contiguous(); scores=decoded[:,4:,:].permute(0,2,1).contiguous(); k=min(max_det,n); aidx=scores.amax(2).topk(k,dim=1).indices
    bsel=boxes.gather(1,aidx.unsqueeze(-1).expand(-1,-1,4)); ssel=scores.gather(1,aidx.unsqueeze(-1).expand(-1,-1,nc)); flat=ssel.reshape(b,-1); k2=min(max_det,flat.shape[1]); vals,fidx=flat.topk(k2,dim=1); local=torch.div(fidx,nc,rounding_mode='floor'); cls=fidx.remainder(nc); orig=aidx.gather(1,local); final_box=xywh2xyxy(bsel.gather(1,local.unsqueeze(-1).expand(-1,-1,4)))
    out=[]
    for bi in range(b):
        keep=vals[bi]>=conf; pred=torch.cat([final_box[bi,keep],vals[bi,keep,None],cls[bi,keep,None].float()],1); out.append({'pred':pred,'anchor':orig[bi,keep].long(),'cls':cls[bi,keep].long()})
    return out

def class_aware_c70(pred,iou=.70):
    if pred.numel()==0:return pred
    keep=nms(pred[:,:4]+pred[:,5:6].repeat(1,4)*7680.0,pred[:,4],iou); return pred[keep]

def component_masks(pred,iou=.70):
    n_pred=int(pred.shape[0]); keep1=torch.ones(n_pred,dtype=torch.bool,device=pred.device); keepall=keep1.clone()
    if n_pred<=1:return keep1,keepall
    classes=pred[:,5].long()
    for cls in classes.unique():
        ids=torch.where(classes==cls)[0]
        if ids.numel()<=1:continue
        adj=(box_iou(pred[ids,:4],pred[ids,:4])>=iou).detach().cpu().numpy(); ids_cpu=ids.detach().cpu().tolist(); visited=set()
        for st in range(len(ids_cpu)):
            if st in visited:continue
            stack=[st]; visited.add(st); comp=[]
            while stack:
                u=stack.pop(); comp.append(u)
                for v in np.where(adj[u])[0].tolist():
                    if v not in visited:visited.add(v);stack.append(v)
            if len(comp)<=1:continue
            glob=[ids_cpu[x] for x in comp]; order=torch.argsort(pred[glob,4],descending=True).detach().cpu().tolist(); ranked=[glob[int(x)] for x in order]; keep1[ranked[1]]=False
            for rid in ranked[1:]:keepall[rid]=False
    return keep1,keepall

def build_selector_maps(loss_fn,selector_iou=.70):
    logits=loss_fn._week48_bce.logits; cap=loss_fn._week48_assigner_last
    if logits is None or cap is None:raise RuntimeError('Capture state missing')
    out=cap['output']; target_bboxes,target_scores,fg_mask=out[1],out[2],out[3].bool(); pd=loss_fn._find_pd_bboxes(cap['inputs'],target_bboxes.shape).detach(); scores=logits.detach(); target_scores=target_scores.detach(); maps=[]
    for bi in range(logits.shape[0]):
        pos=torch.where(fg_mask[bi])[0]; neg=torch.where(~fg_mask[bi])[0]; fg=set(int(x) for x in pos.cpu().tolist()); eligible=set(); selected=set(); epairs=set(); spairs=set()
        if pos.numel()>0 and neg.numel()>0:
            ious=box_iou(pd[bi,pos].float(),pd[bi,neg].float()); pcl=target_scores[bi,pos].argmax(-1).long()
            for pj in range(pos.numel()):
                pa=int(pos[pj]); cls=int(pcl[pj]); cand_local=torch.where(ious[pj]>=selector_iou)[0]
                if cand_local.numel()==0:continue
                cand=neg[cand_local]
                for na_t in cand:
                    na=int(na_t); eligible.add((na,cls)); epairs.add((pa,na,cls))
                hard=int(cand[int(torch.argmax(scores[bi,cand,cls]))]); selected.add((hard,cls)); spairs.add((pa,hard,cls))
        maps.append({'fg_anchors':fg,'eligible_keys':eligible,'selected_keys':selected,'eligible_direct_pairs':epairs,'selected_direct_pairs':spairs,'positive_count':len(fg)})
    return maps

def duplicate_pairs(pred,iou=.70):
    if pred.shape[0]<=1:return []
    m=(box_iou(pred[:,:4],pred[:,:4])>=iou)&pred[:,5].long()[:,None].eq(pred[:,5].long()[None,:])&torch.triu(torch.ones((pred.shape[0],pred.shape[0]),dtype=torch.bool,device=pred.device),diagonal=1)
    return torch.nonzero(m,as_tuple=False).cpu().tolist()

def taxonomy(pred,anchors,classes,smap,pairs):
    fg=smap['fg_anchors']; ep=smap['eligible_direct_pairs']; sp=smap['selected_direct_pairs']; ek=smap['eligible_keys']; sk=smap['selected_keys']; cnt=Counter(); dup=set(); elig_ep=sel_ep=0
    for i,j in pairs:
        ai,aj,c=int(anchors[i]),int(anchors[j]),int(classes[i]); ki,kj=(ai,c),(aj,c); dup.update([ki,kj]); elig_ep+=int(ki in ek or kj in ek); sel_ep+=int(ki in sk or kj in sk)
        if (ai,aj,c) in sp or (aj,ai,c) in sp:cnt['DIRECT_SELECTED']+=1
        elif (ai,aj,c) in ep or (aj,ai,c) in ep:cnt['ELIGIBLE_UNSELECTED']+=1
        elif ai in fg and aj in fg:cnt['POS_POS']+=1
        elif ai not in fg and aj not in fg:cnt['NEG_NEG']+=1
        else:cnt['MIXED_OTHER']+=1
    return {'counts':cnt,'duplicate_keys':len(dup),'selected_duplicate_keys':sum(k in sk for k in dup),'eligible_duplicate_keys':sum(k in ek for k in dup),'eligible_endpoint_pairs':elig_ep,'selected_endpoint_pairs':sel_ep}

def write_preds(writer,pred,path,shape_meta,img_hw,cmap):
    if pred is None or pred.numel()==0:return
    q=pred.clone(); scale_boxes(img_hw,q[:,:4],shape_meta[0],shape_meta[1]); box=xyxy2xywh(q[:,:4]); box[:,:2]-=box[:,2:]/2; image_id=int(Path(path).stem)
    for i in range(q.shape[0]):writer.write_obj({'image_id':image_id,'category_id':int(cmap[int(q[i,5])]),'bbox':[round(float(x),3) for x in box[i].tolist()],'score':round(float(q[i,4]),7)})

def evaluate_one(label,raw_weight,proxy_weight,val_txt,annotation,out_dir,hyp_path,batch=8,imgsz=640,conf=.001,max_det=300,selector_iou=.70,duplicate_iou=.70,workers=2,max_images=0,skip_coco=False):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True); device=torch.device('cuda:0'); proxy=attempt_load(str(proxy_weight),device=device,inplace=True,fuse=False).float().eval(); raw=attempt_load(str(raw_weight),device=device,inplace=True,fuse=False).float(); raw.hyp=dict(yaml.safe_load(Path(hyp_path).read_text())); raw.train()
    for mod in raw.modules():
        if isinstance(mod,torch.nn.modules.batchnorm._BatchNorm):mod.eval()
    if hasattr(raw.model[-1],'week28_dual_training'):raw.model[-1].week28_dual_training=True
    os.environ['YOLO_WEEK48_DUP_HN_IOU']=str(selector_iou); os.environ['YOLO_WEEK48_DUP_HN_LAMBDA']='0.0'; os.environ['YOLO_WEEK48_DUP_HN_TELEMETRY']=''; capture=CaptureLoss(raw); stride=int(proxy.stride.max()); imgsz=int(check_img_size(imgsz,s=stride)); loader=create_dataloader(str(val_txt),imgsz,batch,stride,False,pad=.5,rect=True,workers=workers,prefix=f'week49-{label}: ')[0]
    methods=['A_native_nms','B_nonms_k300','SEL1_exact','SELALL_eligible','COMP1_oracle','COMPALL_oracle','C70_diagnostic']; writers={}; paths={}
    if not skip_coco:
        for m in methods: paths[m]=out_dir/f'{m}.json'; writers[m]=JsonArrayWriter(paths[m])
    cmap=coco80_to_coco91_class(); totals=Counter(); taxcnt=Counter(); align_max=0.; align_means=[]; checked=False
    try:
        for im,targets,impaths,shapes in loader:
            if max_images and totals['images']>=max_images:break
            im=im.to(device).float()/255.; targets=targets.to(device)
            with torch.inference_mode():
                raw_out=raw(im); _=capture(raw_out,targets,img=im,epoch=0); decoded=extract_proxy_decoded(proxy(im))
            smaps=build_selector_maps(capture,selector_iou); logits=capture._week48_bce.logits.detach(); ps=decoded[:,4:,:].permute(0,2,1).contiguous(); diff=(logits.sigmoid()-ps).abs(); align_max=max(align_max,float(diff.max())); align_means.append(float(diff.mean()))
            if not checked:
                if not torch.allclose(logits.sigmoid(),ps,atol=5e-4,rtol=5e-4):raise RuntimeError(f'Anchor alignment failed max={float(diff.max())} mean={float(diff.mean())}')
                checked=True
            indexed=indexed_no_nms(decoded,conf,max_det); A_batch=non_max_suppression(decoded,conf_thres=conf,iou_thres=.70,classes=None,agnostic=False,multi_label=True,labels=(),max_det=max_det)
            for bi in range(len(impaths)):
                if max_images and totals['images']>=max_images:break
                B=indexed[bi]['pred']; anchors=indexed[bi]['anchor']; classes=indexed[bi]['cls']; sm=smaps[bi]; keysB={(int(anchors[k]),int(classes[k])) for k in range(B.shape[0])}; sk,ek=sm['selected_keys'],sm['eligible_keys']; sel1=torch.tensor([(int(anchors[k]),int(classes[k])) not in sk for k in range(B.shape[0])],dtype=torch.bool,device=B.device); selall=torch.tensor([(int(anchors[k]),int(classes[k])) not in ek for k in range(B.shape[0])],dtype=torch.bool,device=B.device); k1,kall=component_masks(B,duplicate_iou); variants={'A_native_nms':A_batch[bi],'B_nonms_k300':B,'SEL1_exact':B[sel1],'SELALL_eligible':B[selall],'COMP1_oracle':B[k1],'COMPALL_oracle':B[kall],'C70_diagnostic':class_aware_c70(B,duplicate_iou)}; pairs=duplicate_pairs(B,duplicate_iou); tx=taxonomy(B,anchors,classes,sm,pairs); totals['images']+=1; totals['predictions_B']+=int(B.shape[0]); totals['duplicate_pairs']+=len(pairs); totals['duplicate_prediction_keys']+=tx['duplicate_keys']; totals['selected_duplicate_keys']+=tx['selected_duplicate_keys']; totals['eligible_duplicate_keys']+=tx['eligible_duplicate_keys']; totals['eligible_endpoint_pairs']+=tx['eligible_endpoint_pairs']; totals['selected_endpoint_pairs']+=tx['selected_endpoint_pairs']; totals['positive_anchors']+=sm['positive_count']; totals['selector_eligible_keys']+=len(ek); totals['selector_selected_keys']+=len(sk); totals['selected_keys_present_B']+=sum(k in keysB for k in sk); totals['eligible_keys_present_B']+=sum(k in keysB for k in ek); taxcnt.update(tx['counts'])
                if not skip_coco:
                    for m,p in variants.items():write_preds(writers[m],p,impaths[bi],shapes[bi],im.shape[2:],cmap)
    finally:
        for w in writers.values():w.close()
    total_pairs=totals['duplicate_pairs']; taxfrac={k:(v/total_pairs if total_pairs else 0.) for k,v in sorted(taxcnt.items())}; selector={'positive_anchors_total':totals['positive_anchors'],'eligible_keys_total':totals['selector_eligible_keys'],'selected_keys_total':totals['selector_selected_keys'],'selected_keys_present_in_B':totals['selected_keys_present_B'],'eligible_keys_present_in_B':totals['eligible_keys_present_B'],'selected_key_top300_presence_rate':totals['selected_keys_present_B']/totals['selector_selected_keys'] if totals['selector_selected_keys'] else 0.,'eligible_key_top300_presence_rate':totals['eligible_keys_present_B']/totals['selector_eligible_keys'] if totals['selector_eligible_keys'] else 0.}; dt={'sameclass_duplicate_pairs_iou70':total_pairs,'sameclass_duplicate_pairs_iou70_per_image':total_pairs/totals['images'],'category_counts':dict(sorted(taxcnt.items())),'category_fraction':taxfrac,'duplicate_prediction_keys_total':totals['duplicate_prediction_keys'],'selected_duplicate_keys_total':totals['selected_duplicate_keys'],'eligible_duplicate_keys_total':totals['eligible_duplicate_keys'],'selected_duplicate_key_coverage':totals['selected_duplicate_keys']/totals['duplicate_prediction_keys'] if totals['duplicate_prediction_keys'] else 0.,'eligible_duplicate_key_coverage':totals['eligible_duplicate_keys']/totals['duplicate_prediction_keys'] if totals['duplicate_prediction_keys'] else 0.,'selected_endpoint_pair_coverage':totals['selected_endpoint_pairs']/total_pairs if total_pairs else 0.,'eligible_endpoint_pair_coverage':totals['eligible_endpoint_pairs']/total_pairs if total_pairs else 0.}; methodout={}
    if not skip_coco:
        for m in methods:methodout[m]={'prediction_count':writers[m].count,'predictions_per_image':writers[m].count/totals['images'],'coco_api':coco_eval(annotation,paths[m]),'prediction_json':str(paths[m])}
    result={'label':label,'raw_weight':str(raw_weight),'proxy_weight':str(proxy_weight),'images':totals['images'],'config':{'batch':batch,'imgsz':imgsz,'conf':conf,'max_det':max_det,'selector_iou':selector_iou,'duplicate_iou':duplicate_iou,'precision':'FP32','training_performed':False},'anchor_alignment':{'raw_o2o_sigmoid_vs_proxy_score_max_abs':align_max,'raw_o2o_sigmoid_vs_proxy_score_mean_abs':float(np.mean(align_means)),'tolerance_atol_rtol':5e-4,'pass':checked},'selector':selector,'duplicate_taxonomy':dt,'methods':methodout}; (out_dir/'week49_diagnostic.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)); del raw,proxy,capture; torch.cuda.empty_cache(); return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--label',required=True); ap.add_argument('--raw-weight',required=True); ap.add_argument('--proxy-weight',required=True); ap.add_argument('--val',required=True); ap.add_argument('--annotation',required=True); ap.add_argument('--out-dir',required=True); ap.add_argument('--hyp',required=True); ap.add_argument('--batch',type=int,default=8); ap.add_argument('--imgsz',type=int,default=640); ap.add_argument('--conf',type=float,default=.001); ap.add_argument('--max-det',type=int,default=300); ap.add_argument('--selector-iou',type=float,default=.70); ap.add_argument('--duplicate-iou',type=float,default=.70); ap.add_argument('--workers',type=int,default=2); ap.add_argument('--max-images',type=int,default=0); ap.add_argument('--skip-coco',action='store_true'); a=ap.parse_args(); r=evaluate_one(a.label,Path(a.raw_weight),Path(a.proxy_weight),Path(a.val),Path(a.annotation),Path(a.out_dir),Path(a.hyp),a.batch,a.imgsz,a.conf,a.max_det,a.selector_iou,a.duplicate_iou,a.workers,a.max_images,a.skip_coco); print(json.dumps({'label':r['label'],'images':r['images'],'anchor_alignment':r['anchor_alignment'],'selector':r['selector'],'duplicate_taxonomy':r['duplicate_taxonomy'],'method_ap':{k:v['coco_api']['ap50_95'] for k,v in r['methods'].items()} if r['methods'] else {}},indent=2,ensure_ascii=False))
if __name__=='__main__':main()
