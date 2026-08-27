from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.dataloaders import create_dataloader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    data = yaml.safe_load(Path(a.data).read_text(encoding="utf-8"))
    hyp = yaml.safe_load(Path(a.hyp).read_text(encoding="utf-8"))
    train_path = Path(data["train"])
    labels_dir = Path(str(train_path).replace("/images/", "/labels/"))
    cache_path = labels_dir.with_suffix(".cache")
    cache_path.unlink(missing_ok=True)

    loader, dataset = create_dataloader(
        str(train_path), 640, a.batch, 32,
        single_cls=False, hyp=hyp, augment=True, cache=False,
        rect=False, rank=-1, workers=2, image_weights=False,
        close_mosaic=False, quad=False, min_items=0,
        prefix="week28v2-train-gate: ", shuffle=True,
    )

    if not cache_path.exists():
        raise RuntimeError(f"Expected cache missing: {cache_path}")

    cache = np.load(cache_path, allow_pickle=True).item()
    results = cache["results"]
    nf, nm, ne, nc, total = map(int, results)
    metadata = {"hash", "results", "msgs", "version"}
    entries = len([k for k in cache.keys() if k not in metadata])

    report = {
        "dataset_len": len(dataset),
        "loader_len": len(loader),
        "batch": a.batch,
        "cache_path": str(cache_path),
        "cache_entries": entries,
        "cache_found": nf,
        "cache_missing": nm,
        "cache_empty": ne,
        "cache_corrupt": nc,
        "cache_total": total,
    }
    report["pass"] = bool(
        len(dataset) == 5000 and len(loader) == 625
        and entries == 5000 and nf == 5000 and nm == 0
        and ne == 0 and nc == 0 and total == 5000
    )

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise RuntimeError("Week28 v2 dataloader hard gate FAIL; training forbidden")


if __name__ == "__main__":
    main()
