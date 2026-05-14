from pathlib import Path
import sys
import argparse
import json
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from models.yolo import DetectionModel
from models.common import DetectMultiBackend
from utils.e2e_postprocess import (
    v10_no_nms_postprocess,
    class_aware_nms_postprocess,
    profile_function,
    select_one2one,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--cfg", type=str, default="models/detect/yolov9-s-v10dual.yaml")
    parser.add_argument("--img", type=int, default=320)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--json", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    if args.weights:
        model = DetectMultiBackend(args.weights, device=device)
        model.eval()
        if hasattr(model, "model") and hasattr(model.model, "eval"):
            model.model.eval()
    else:
        model = DetectionModel(args.cfg, ch=3, nc=80).to(device).eval()

    im = torch.zeros(args.batch, 3, args.img, args.img, device=device)

    with torch.no_grad():
        for _ in range(args.warmup):
            preds = model(im)

        if isinstance(preds, dict):
            if hasattr(model, "model") and hasattr(model.model, "eval"):
                model.model.eval()
            model.eval()
            preds = model(im)

        if device.type == "cuda":
            torch.cuda.synchronize()

        s = time.perf_counter()
        for _ in range(args.repeat):
            preds = model(im)
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_ms = (time.perf_counter() - s) * 1000.0 / args.repeat

        if isinstance(preds, dict):
            raise TypeError(
                "benchmark received dict output after eval(). "
                "Check V10DualDDetect forward() and checkpoint loading mode."
            )

        one2one = select_one2one(preds)

        no_nms_ms = profile_function(
            lambda p: v10_no_nms_postprocess(p, conf_thres=args.conf, max_det=args.max_det),
            preds,
            repeat=args.repeat,
            warmup=args.warmup,
        )
        nms_ms = profile_function(
            lambda p: class_aware_nms_postprocess(p, conf_thres=args.conf, iou_thres=args.iou, max_det=args.max_det),
            preds,
            repeat=args.repeat,
            warmup=args.warmup,
        )

    result = {
        "device": str(device),
        "weights": args.weights,
        "cfg": args.cfg,
        "img": args.img,
        "batch": args.batch,
        "repeat": args.repeat,
        "one2one_shape": list(one2one.shape),
        "forward_ms": forward_ms,
        "no_nms_postprocess_ms": no_nms_ms,
        "nms_postprocess_ms": nms_ms,
        "postprocess_speedup_nms_over_no_nms": nms_ms / max(no_nms_ms, 1e-12),
    }

    print(json.dumps(result, indent=2))

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
