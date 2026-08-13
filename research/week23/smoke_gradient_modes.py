
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


def formula_gradient_test():
    print("=== Synthetic gradient-direction test ===")

    for mode in ["pairwise", "positive_only", "negative_only"]:
        z_pos = torch.tensor(0.2, requires_grad=True)
        z_neg = torch.tensor(0.3, requires_grad=True)

        if mode == "positive_only":
            pos_for_loss = z_pos
            neg_for_loss = z_neg.detach()
        elif mode == "negative_only":
            pos_for_loss = z_pos.detach()
            neg_for_loss = z_neg
        else:
            pos_for_loss = z_pos
            neg_for_loss = z_neg

        loss = torch.nn.functional.softplus(
            neg_for_loss - pos_for_loss
        )
        loss.backward()

        gp = 0.0 if z_pos.grad is None else float(z_pos.grad.item())
        gn = 0.0 if z_neg.grad is None else float(z_neg.grad.item())

        print(mode, {
            "loss": float(loss.item()),
            "grad_pos": gp,
            "grad_neg": gn,
        })

        if mode == "pairwise":
            assert gp < 0 and gn > 0
        elif mode == "positive_only":
            assert gp < 0 and abs(gn) < 1e-12
        else:
            assert abs(gp) < 1e-12 and gn > 0

    print("Synthetic gradient-direction test passed.")


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
        "prefix": "week23-smoke: ",
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


def full_model_test(
    weights,
    data,
    imgsz,
    batch,
    workers,
    rank_weight,
    rank_iou,
):
    print("\n=== Full-model mode smoke ===")

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    model = attempt_load(weights, device=device)
    model.train()

    stride = int(
        model.stride.max().item()
        if torch.is_tensor(model.stride)
        else max(model.stride)
    )

    loader = make_loader(
        resolve_dataset_path(data),
        imgsz,
        batch,
        stride,
        workers,
    )

    batch_data = next(iter(loader))
    imgs = batch_data[0].to(device).float() / 255.0
    targets = batch_data[1].to(device)

    for mode in ["pairwise", "positive_only", "negative_only"]:
        model.zero_grad(set_to_none=True)

        os.environ["YOLO_REP_RANK_WEIGHT"] = str(rank_weight)
        os.environ["YOLO_REP_RANK_IOU"] = str(rank_iou)
        os.environ["YOLO_REP_RANK_MODE"] = mode
        os.environ["YOLO_REP_RANK_TELEMETRY"] = ""

        criterion = ComputeLoss(model)
        assert criterion.rep_rank_mode == mode

        outputs = model(imgs)
        loss, _ = criterion(
            outputs,
            targets,
            img=imgs,
        )

        raw = getattr(
            criterion,
            "_week22_last_rep_rank_raw",
            None,
        )

        stats = getattr(
            criterion,
            "_week22_last_rep_rank_stats",
            None,
        )

        assert torch.isfinite(loss)
        assert raw is not None and torch.isfinite(raw)
        assert stats is not None
        assert stats["active_positive_count"] > 0

        loss.backward()

        grad_sum = 0.0
        grad_count = 0

        for p in model.parameters():
            if p.grad is not None:
                grad_sum += float(
                    p.grad.detach().abs().sum().cpu().item()
                )
                grad_count += 1

        assert grad_count > 0
        assert grad_sum > 0

        print({
            "mode": mode,
            "total_loss": float(loss.detach().cpu().item()),
            "rank_loss_raw": float(raw.detach().cpu().item()),
            "active_pairs": stats["active_positive_count"],
            "winner_rate_before": stats["winner_rate_before"],
            "grad_abs_sum": grad_sum,
        })

    print("\nFull-model mode smoke passed.")


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

    formula_gradient_test()

    full_model_test(
        weights=args.weights,
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        rank_weight=args.rank_weight,
        rank_iou=args.rank_iou,
    )


if __name__ == "__main__":
    main()
