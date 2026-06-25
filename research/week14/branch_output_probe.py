
import argparse, csv, json, sys
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes

try:
    from utils.e2e_postprocess import select_one2one, v10_no_nms_postprocess, class_aware_nms_postprocess
except Exception as e:
    select_one2one = None
    v10_no_nms_postprocess = None
    class_aware_nms_postprocess = None
    IMPORT_ERROR = repr(e)
else:
    IMPORT_ERROR = None

def load_img(path, imgsz, stride=32):
    im0 = cv2.imread(str(path))
    if im0 is None:
        raise FileNotFoundError(path)
    im = letterbox(im0, imgsz, stride=stride, auto=True)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).float() / 255.0
    if im.ndim == 3:
        im = im.unsqueeze(0)
    return im, im0

def summary_obj(x):
    if torch.is_tensor(x):
        return {"type": "tensor", "shape": list(x.shape), "mean": float(x.float().mean()), "min": float(x.min()), "max": float(x.max())}
    if isinstance(x, (list, tuple)):
        return {"type": type(x).__name__, "len": len(x), "items": [summary_obj(v) for v in x[:10]]}
    if isinstance(x, dict):
        return {"type": "dict", "keys": list(x.keys())}
    return {"type": str(type(x))}

def as_det(x):
    if x is None:
        return torch.empty((0, 6))
    if torch.is_tensor(x):
        return x[0] if x.ndim == 3 else x
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return torch.empty((0, 6))
        if torch.is_tensor(x[0]):
            t = x[0]
            return t[0] if t.ndim == 3 else t
        for v in x:
            t = as_det(v)
            if t is not None:
                return t
    return torch.empty((0, 6))

def postprocess(raw, mode, conf, iou, max_det):
    if mode == "native_nms":
        pred = raw[0] if isinstance(raw, (list, tuple)) else raw
        return non_max_suppression(pred, conf_thres=conf, iou_thres=iou, max_det=max_det)[0]
    if IMPORT_ERROR:
        raise RuntimeError(IMPORT_ERROR)
    one = select_one2one(raw)
    if mode == "one2one_nms":
        return as_det(class_aware_nms_postprocess(one, conf_thres=conf, iou_thres=iou, max_det=max_det))
    if mode == "one2one_no_nms":
        return as_det(v10_no_nms_postprocess(one, conf_thres=conf, iou_thres=iou, max_det=max_det))
    raise ValueError(mode)

def det_stats(det):
    if det is None or len(det) == 0:
        return {"num_pred": 0, "conf_min": None, "conf_mean": None, "conf_max": None}
    det = det.detach().cpu()
    conf = det[:, 4].float().numpy()
    return {"num_pred": int(len(det)), "conf_min": float(conf.min()), "conf_mean": float(conf.mean()), "conf_max": float(conf.max())}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--sample-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--sample-n", type=int, default=16)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = attempt_load(args.weights, device=device)
    model.eval()

    stride = 32
    if hasattr(model, "stride"):
        try:
            stride = int(model.stride.max())
        except Exception:
            pass

    imgs = sorted((Path(args.sample_root) / "images" / "train").glob("*.jpg"))[:args.sample_n]
    assert imgs, "no sample images"

    rows, structures = [], []
    with torch.no_grad():
        for img in imgs:
            im, im0 = load_img(img, args.imgsz, stride)
            im = im.to(device)
            raw = model(im)
            structures.append({"image": img.name, "raw_summary": summary_obj(raw)})
            for mode in ["native_nms", "one2one_nms", "one2one_no_nms"]:
                try:
                    det = postprocess(raw, mode, args.conf, args.iou, args.max_det)
                    if det is not None and len(det):
                        det = det.detach().clone()
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    row = {"image": img.name, "mode": mode, "error": ""}
                    row.update(det_stats(det))
                except Exception as e:
                    row = {"image": img.name, "mode": mode, "num_pred": None, "conf_min": None, "conf_mean": None, "conf_max": None, "error": repr(e)}
                rows.append(row)

    with (out_dir / "branch_prediction_probe.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "mode", "num_pred", "conf_min", "conf_mean", "conf_max", "error"])
        w.writeheader()
        w.writerows(rows)

    (out_dir / "raw_output_structure.json").write_text(json.dumps(structures, indent=2), encoding="utf-8")

    import pandas as pd
    df = pd.DataFrame(rows)
    agg = df.groupby("mode").agg(
        images=("image", "count"),
        mean_num_pred=("num_pred", "mean"),
        median_num_pred=("num_pred", "median"),
        mean_conf=("conf_mean", "mean"),
        max_conf=("conf_max", "max"),
    ).reset_index()
    agg.to_csv(out_dir / "branch_prediction_probe_aggregate.csv", index=False)
    print(agg)

if __name__ == "__main__":
    main()
