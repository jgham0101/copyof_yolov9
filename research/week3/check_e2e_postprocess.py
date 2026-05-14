from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from models.yolo import DetectionModel, V10DualDDetect
from utils.e2e_postprocess import (
    v10_no_nms_postprocess,
    class_aware_nms_postprocess,
    profile_function,
    select_one2one,
)


def check_synthetic():
    pred = torch.rand(2, 84, 1000)
    pred[:, :4, :] *= 320
    pred[:, 4:, :] *= 0.2
    pred[:, 10, :20] = 0.99

    no_nms = v10_no_nms_postprocess(pred, conf_thres=0.01, max_det=50)
    nms = class_aware_nms_postprocess(pred, conf_thres=0.01, iou_thres=0.7, max_det=50)

    assert len(no_nms) == 2
    assert len(nms) == 2
    assert all(x.ndim == 2 and x.shape[1] == 6 for x in no_nms)
    assert all(x.shape[0] <= 50 for x in no_nms)
    assert all(x.shape[0] <= 50 for x in nms)


def check_model():
    cfg = ROOT / "models/detect/yolov9-s-v10dual.yaml"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = DetectionModel(str(cfg), ch=3, nc=80).to(device).eval()
    assert isinstance(model.model[-1], V10DualDDetect)

    x = torch.zeros(1, 3, 256, 256, device=device)
    with torch.no_grad():
        preds = model(x)

    one2one = select_one2one(preds)
    assert one2one.ndim == 3

    det = v10_no_nms_postprocess(preds, conf_thres=0.001, max_det=100)
    assert len(det) == 1
    assert det[0].shape[1] == 6

    no_nms_ms = profile_function(
        lambda p: v10_no_nms_postprocess(p, conf_thres=0.001, max_det=100),
        preds,
        repeat=20,
        warmup=5,
    )
    nms_ms = profile_function(
        lambda p: class_aware_nms_postprocess(p, conf_thres=0.001, iou_thres=0.7, max_det=100),
        preds,
        repeat=20,
        warmup=5,
    )

    print("Week 3 no-NMS postprocess smoke test passed")
    print("device:", device)
    print("one2one shape:", tuple(one2one.shape))
    print("detections:", tuple(det[0].shape))
    print(f"no_nms_ms: {no_nms_ms:.4f}")
    print(f"nms_ms    : {nms_ms:.4f}")


if __name__ == "__main__":
    check_synthetic()
    check_model()
