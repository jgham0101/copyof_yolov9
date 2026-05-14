from pathlib import Path
import argparse
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import torch

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import check_img_size, increment_path, colorstr
try:
    from utils.general import scale_boxes
except Exception:
    from utils.general import scale_coords as scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.e2e_postprocess import v10_no_nms_postprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, required=True)
    parser.add_argument("--source", type=str, default="data/images")
    parser.add_argument("--imgsz", "--img", nargs="+", type=int, default=[640])
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="e2e")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--hide-labels", action="store_true")
    parser.add_argument("--hide-conf", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main(opt):
    source = str(opt.source)
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(opt.device)
    model = DetectMultiBackend(opt.weights, device=device)
    model.eval()
    if hasattr(model, "model") and hasattr(model.model, "eval"):
        model.model.eval()
    stride, names, pt = model.stride, model.names, model.pt

    imgsz = opt.imgsz * 2 if len(opt.imgsz) == 1 else opt.imgsz
    imgsz = check_img_size(imgsz, s=stride)

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    model.warmup(imgsz=(1, 3, *imgsz))

    seen = 0
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.float() / 255.0
        if im.ndim == 3:
            im = im[None]

        preds = model(im)
        dets = v10_no_nms_postprocess(preds, conf_thres=opt.conf_thres, max_det=opt.max_det)

        for det in dets:
            seen += 1
            p = Path(path)
            im0 = im0s.copy()
            annotator = Annotator(im0, line_width=2, example=str(names))

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = None if opt.hide_labels else (
                        names[c] if opt.hide_conf else f"{names[c]} {conf:.2f}"
                    )
                    annotator.box_label(xyxy, label, color=colors(c, True))

            save_path = str(save_dir / p.name)
            cv2.imwrite(save_path, annotator.result())
            print(f"{s} saved to {save_path}")

    print(f"No-NMS detection complete. {seen} image(s), results saved to {colorstr('bold', save_dir)}")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
