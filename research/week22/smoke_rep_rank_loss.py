
import argparse
import inspect
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.experimental import attempt_load
from utils.dataloaders import create_dataloader
from utils.loss_tal_dual import ComputeLoss


def resolve_dataset_path(data_yaml):
    data = yaml.safe_load(
        Path(data_yaml).read_text(encoding="utf-8")
    )

    root = Path(data.get("path", ""))
    train = data.get("train")

    if isinstance(train, list):
        train = train[0]

    p = Path(train)

    if not p.is_absolute():
        p = root / p

    return str(p)


def make_loader(path, imgsz, batch, stride, workers):
    sig = inspect.signature(create_dataloader)

    optional = {
        "single_cls": False,
        "hyp": None,
        "augment": False,
        "cache": False,
        "pad": 0.0,
        "rect": False,
        "rank": -1,
        "workers": workers,
        "image_weights": False,
        "quad": False,
        "prefix": "week22-smoke: ",
        "shuffle": False,
        "seed": 0,
    }

    kwargs = {
        k: v
        for k, v in optional.items()
        if k in sig.parameters
    }

    loader, _ = create_dataloader(
        path,
        imgsz,
        batch,
        stride,
        **kwargs,
    )

    return loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rank-weight", type=float, default=0.1)
    ap.add_argument("--rank-iou", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    model = attempt_load(
        args.weights,
        device=device,
    )

    model.train()

    stride = int(
        model.stride.max().item()
        if torch.is_tensor(model.stride)
        else max(model.stride)
    )

    loader = make_loader(
        resolve_dataset_path(args.data),
        args.imgsz,
        args.batch,
        stride,
        args.workers,
    )

    batch = next(iter(loader))

    imgs = (
        batch[0]
        .to(device)
        .float()
        / 255.0
    )

    targets = batch[1].to(device)

    outputs = model(imgs)

    os.environ["YOLO_REP_RANK_IOU"] = str(args.rank_iou)
    os.environ["YOLO_REP_RANK_TELEMETRY"] = ""

    os.environ["YOLO_REP_RANK_WEIGHT"] = "0.0"
    criterion_base = ComputeLoss(model)
    loss_base, _ = criterion_base(
        outputs,
        targets,
        img=imgs,
    )

    os.environ["YOLO_REP_RANK_WEIGHT"] = str(args.rank_weight)
    criterion_rank = ComputeLoss(model)
    loss_rank, _ = criterion_rank(
        outputs,
        targets,
        img=imgs,
    )

    raw = getattr(
        criterion_rank,
        "_week22_last_rep_rank_raw",
        None,
    )

    stats = getattr(
        criterion_rank,
        "_week22_last_rep_rank_stats",
        None,
    )

    print("loss_base:", float(loss_base.detach().cpu().item()))
    print("loss_rank:", float(loss_rank.detach().cpu().item()))
    print(
        "rank_raw:",
        float(raw.detach().cpu().item())
        if torch.is_tensor(raw)
        else raw,
    )
    print("rank_stats:", stats)

    assert torch.isfinite(loss_base)
    assert torch.isfinite(loss_rank)
    assert raw is not None
    assert torch.isfinite(raw)
    assert stats is not None
    assert stats["active_positive_count"] > 0
    assert float(loss_rank.detach().cpu().item()) > float(
        loss_base.detach().cpu().item()
    )

    model.zero_grad(set_to_none=True)
    loss_rank.backward()

    grad_sum = 0.0
    grad_count = 0

    for p in model.parameters():
        if p.grad is not None:
            grad_sum += float(
                p.grad.detach().abs().sum().cpu().item()
            )
            grad_count += 1

    print("parameters with grad:", grad_count)
    print("grad_abs_sum:", grad_sum)

    assert grad_count > 0
    assert grad_sum > 0

    print("\nWeek22 ranking-loss smoke test passed.")


if __name__ == "__main__":
    main()
