from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.common import DetectMultiBackend
from utils.dataloaders import create_dataloader
from utils.e2e_postprocess import v10_no_nms_postprocess, class_aware_nms_postprocess
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    check_dataset,
    check_img_size,
    check_yaml,
    colorstr,
    increment_path,
    print_args,
    xywh2xyxy,
)
try:
    from utils.general import scale_boxes
except Exception:
    from utils.general import scale_coords as scale_boxes
from utils.metrics import ap_per_class, box_iou
from utils.torch_utils import select_device, smart_inference_mode


def process_batch(detections: torch.Tensor, labels: torch.Tensor, iouv: torch.Tensor) -> torch.Tensor:
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    if detections.shape[0] == 0 or labels.shape[0] == 0:
        return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]

    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True

    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode()
def run(
    data,
    weights,
    batch_size=4,
    imgsz=320,
    conf_thres=0.001,
    iou_thres=0.7,
    max_det=300,
    task='val',
    device='',
    workers=2,
    single_cls=False,
    half=False,
    dnn=False,
    postprocess='no-nms',
    project=ROOT / 'runs/val',
    name='e2e',
    exist_ok=False,
    min_items=0,
):
    device = select_device(device, batch_size=batch_size)
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    data_dict = check_dataset(data)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    model.eval()
    if hasattr(model, 'model') and hasattr(model.model, 'eval'):
        model.model.eval()

    stride, pt = model.stride, model.pt
    imgsz = check_img_size(imgsz, s=stride)
    half = model.fp16

    if not pt:
        batch_size = 1

    nc = 1 if single_cls else int(data_dict['nc'])
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()

    dataloader = create_dataloader(
        data_dict[task], imgsz, batch_size, stride, single_cls,
        pad=0.5, rect=pt, workers=workers, min_items=min_items,
        prefix=colorstr(f'{task}: ')
    )[0]

    names = model.names if hasattr(model, 'names') else model.module.names
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))

    seen = 0
    stats = []
    dt = Profile(), Profile(), Profile()
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')

    for batch_i, (im, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)):
        with dt[0]:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
            im = im.half() if half else im.float()
            im /= 255
            nb, _, height, width = im.shape

        with dt[1]:
            preds = model(im)

        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)

        with dt[2]:
            if postprocess == 'no-nms':
                preds = v10_no_nms_postprocess(preds, conf_thres=conf_thres, max_det=max_det)
            elif postprocess == 'nms':
                preds = class_aware_nms_postprocess(preds, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det)
            else:
                raise ValueError(f'unsupported postprocess mode: {postprocess}')

        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                continue

            if single_cls:
                pred[:, 5] = 0

            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])

            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                correct = process_batch(predn, labelsn, iouv)

            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))

    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)] if stats else []
    mp = mr = map50 = map_ = 0.0
    nt = np.zeros(nc)

    if len(stats):
        nt = np.bincount(stats[3].astype(int), minlength=nc)
        if stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir=save_dir, names=names)
            ap50, ap_mean = ap[:, 0], ap.mean(1)
            mp, mr, map50, map_ = p.mean(), r.mean(), ap50.mean(), ap_mean.mean()

    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4
    LOGGER.info(pf % ('all', seen, int(nt.sum()), mp, mr, map50, map_))

    speed = tuple(x.t / max(seen, 1) * 1e3 for x in dt)
    LOGGER.info(
        f'Speed: {speed[0]:.3f}ms pre-process, {speed[1]:.3f}ms inference, '
        f'{speed[2]:.3f}ms {postprocess} postprocess per image at shape {(batch_size, 3, imgsz, imgsz)}'
    )

    result = {
        'postprocess': postprocess,
        'weights': str(weights[0] if isinstance(weights, list) else weights),
        'images': int(seen),
        'instances': int(nt.sum()) if len(nt) else 0,
        'precision': float(mp),
        'recall': float(mr),
        'map50': float(map50),
        'map50_95': float(map_),
        'preprocess_ms': float(speed[0]),
        'inference_ms': float(speed[1]),
        'postprocess_ms': float(speed[2]),
        'imgsz': int(imgsz),
        'batch_size': int(batch_size),
        'conf_thres': float(conf_thres),
        'iou_thres': float(iou_thres),
        'max_det': int(max_det),
    }

    result_path = save_dir / 'metrics.json'
    result_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    LOGGER.info(f'Metrics saved to {colorstr("bold", result_path)}')
    LOGGER.info(f'Results saved to {colorstr("bold", save_dir)}')
    return result


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco.yaml')
    parser.add_argument('--weights', nargs='+', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=320)
    parser.add_argument('--conf-thres', type=float, default=0.001)
    parser.add_argument('--iou-thres', type=float, default=0.7)
    parser.add_argument('--max-det', type=int, default=300)
    parser.add_argument('--task', default='val')
    parser.add_argument('--device', default='')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--single-cls', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--dnn', action='store_true')
    parser.add_argument('--postprocess', choices=['no-nms', 'nms'], default='no-nms')
    parser.add_argument('--project', default=ROOT / 'runs/val')
    parser.add_argument('--name', default='e2e')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--min-items', type=int, default=0)
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)
    print_args(vars(opt))
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    run(**vars(opt))
