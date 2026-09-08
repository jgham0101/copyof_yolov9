from __future__ import annotations

import json
import os
from pathlib import Path
from utils.loss_tal import ComputeLoss as NativeComputeLoss
from utils.loss_tal_week39_o2o_cls2_native import ComputeLoss as O2OCls2ComputeLoss


def native_loss_with_env(model, topk, alpha=0.5, beta=6.0, loss_cls=NativeComputeLoss):
    old = {k: os.environ.get(k) for k in ["YOLOM", "YOLOA", "YOLOB"]}
    os.environ["YOLOM"] = str(topk)
    os.environ["YOLOA"] = str(alpha)
    os.environ["YOLOB"] = str(beta)
    try:
        return loss_cls(model)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class ComputeLossWeek28V2:
    def __init__(self, model):
        self.o2m_topk = 10
        self.o2o_topk = 1
        self.alpha = 0.5
        self.beta = 6.0
        self.o2m = native_loss_with_env(model, 10, self.alpha, self.beta)
        self.o2o = native_loss_with_env(model, 1, self.alpha, self.beta, loss_cls=O2OCls2ComputeLoss)
        self.o2o_weight = float(os.getenv("YOLO_O2O_LOSS_WEIGHT", "1.0"))
        self.telemetry_path = os.getenv("YOLO_WEEK28_V2_TELEMETRY", "")
        self.telemetry_every = int(os.getenv("YOLO_WEEK28_V2_TELEMETRY_EVERY", "50"))
        self.call_index = 0

    def _append(self, row):
        if not self.telemetry_path:
            return
        p = Path(self.telemetry_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def record_clip_norm(self, grad_norm, ni, epoch):
        g = float(grad_norm)
        self._append({
            "type": "clip", "ni": int(ni), "epoch": int(epoch),
            "grad_norm_before_clip": g, "max_norm": 10.0,
            "clip_coefficient": min(1.0, 10.0/(g+1e-12)),
            "o2o_weight": self.o2o_weight,
        })

    def __call__(self, p, targets, img=None, epoch=0):
        if not isinstance(p, dict) or set(p) != {"o2m", "o2o"}:
            raise TypeError("Week28 v2 expects {'o2m','o2o'} training output")
        lo2m, io2m = self.o2m(p["o2m"], targets, img=img, epoch=epoch)
        lo2o, io2o = self.o2o(p["o2o"], targets, img=img, epoch=epoch)
        total = lo2m + self.o2o_weight * lo2o
        items = io2m + self.o2o_weight * io2o
        self.call_index += 1
        if self.call_index <= 5 or self.call_index % self.telemetry_every == 0:
            self._append({
                "type": "loss", "call_index": self.call_index,
                "o2o_weight": self.o2o_weight,
                "o2m_topk": self.o2m_topk, "o2o_topk": self.o2o_topk,
                "alpha": self.alpha, "beta": self.beta,
                "o2m_box": float(io2m[0].item()), "o2m_cls": float(io2m[1].item()), "o2m_dfl": float(io2m[2].item()),
                "o2o_box": float(io2o[0].item()), "o2o_cls": float(io2o[1].item()), "o2o_dfl": float(io2o[2].item()),
                "combined_box": float(items[0].item()), "combined_cls": float(items[1].item()), "combined_dfl": float(items[2].item()),
            })
        return total, items
