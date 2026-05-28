from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import random
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import yaml
from tqdm import tqdm
from pycocotools.coco import COCO


ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download_file(url: str, dst: Path, chunk_size: int = 1024 * 1024):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"exists: {dst}")
        return

    print(f"downloading: {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with dst.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def prepare_annotations(root: Path) -> Path:
    ann_dir = root / "annotations"
    train_json = ann_dir / "instances_train2017.json"
    val_json = ann_dir / "instances_val2017.json"

    if train_json.exists() and val_json.exists():
        print("annotations already prepared")
        return ann_dir

    zip_path = root / "annotations_trainval2017.zip"
    download_file(ANN_URL, zip_path)

    print("extracting annotations...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)

    assert train_json.exists(), train_json
    assert val_json.exists(), val_json
    return ann_dir


def valid_ann(ann):
    if ann.get("iscrowd", 0):
        return False
    x, y, w, h = ann["bbox"]
    return w > 1 and h > 1 and ann.get("area", 0) > 1


def select_images(coco: COCO, target: int, min_per_class: int, seed: int):
    rng = random.Random(seed)

    cat_ids = sorted(coco.getCatIds())
    eligible_img_ids = set()
    cat_to_img_ids = {}

    for cat_id in cat_ids:
        ann_ids = coco.getAnnIds(catIds=[cat_id], iscrowd=False)
        img_ids = set()
        for ann in coco.loadAnns(ann_ids):
            if valid_ann(ann):
                img_ids.add(ann["image_id"])
                eligible_img_ids.add(ann["image_id"])
        ids = list(img_ids)
        rng.shuffle(ids)
        cat_to_img_ids[cat_id] = ids

    selected = []
    selected_set = set()

    # balanced first-pass
    for cat_id in cat_ids:
        added = 0
        for img_id in cat_to_img_ids[cat_id]:
            if img_id not in selected_set:
                selected.append(img_id)
                selected_set.add(img_id)
                added += 1
                if len(selected) >= target:
                    return selected
            if added >= min_per_class:
                break

    # fill remaining randomly
    remaining = list(eligible_img_ids - selected_set)
    rng.shuffle(remaining)
    for img_id in remaining:
        selected.append(img_id)
        selected_set.add(img_id)
        if len(selected) >= target:
            break

    if len(selected) < target:
        print(f"warning: selected {len(selected)} images, target was {target}")

    return selected


def download_image(img: dict, split_name: str, dst_dir: Path, retries: int = 3):
    dst = dst_dir / img["file_name"]
    if dst.exists() and dst.stat().st_size > 0:
        return True, str(dst), "exists"

    url = img.get("coco_url")
    if not url:
        url = f"http://images.cocodataset.org/{split_name}/{img['file_name']}"

    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            dst.write_bytes(r.content)
            if dst.exists() and dst.stat().st_size > 0:
                return True, str(dst), "downloaded"
        except Exception as e:
            last = repr(e)

    return False, str(dst), last


def convert_labels(coco: COCO, selected_ids: list[int], out_label_dir: Path, cat_id_to_idx: dict[int, int]):
    out_label_dir.mkdir(parents=True, exist_ok=True)
    ann_count = 0

    for img_id in selected_ids:
        img = coco.loadImgs([img_id])[0]
        w_img, h_img = img["width"], img["height"]
        label_path = out_label_dir / (Path(img["file_name"]).stem + ".txt")

        lines = []
        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            if not valid_ann(ann):
                continue

            x, y, w, h = ann["bbox"]
            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(w_img), x + w)
            y2 = min(float(h_img), y + h)
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            if bw <= 1 or bh <= 1:
                continue

            cx = (x1 + x2) / 2.0 / w_img
            cy = (y1 + y2) / 2.0 / h_img
            nw = bw / w_img
            nh = bh / h_img

            cls = cat_id_to_idx[ann["category_id"]]
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            ann_count += 1

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return ann_count


def write_stats(coco: COCO, selected_ids: list[int], cat_id_to_idx: dict[int, int], out_csv: Path):
    rows = []
    selected_set = set(selected_ids)

    for cat in sorted(coco.loadCats(coco.getCatIds()), key=lambda x: x["id"]):
        cat_id = cat["id"]
        img_ids = set()
        inst = 0
        for ann in coco.loadAnns(coco.getAnnIds(catIds=[cat_id], iscrowd=False)):
            if ann["image_id"] in selected_set and valid_ann(ann):
                img_ids.add(ann["image_id"])
                inst += 1
        rows.append({
            "class_index": cat_id_to_idx[cat_id],
            "category_id": cat_id,
            "name": cat["name"],
            "num_images": len(img_ids),
            "num_instances": inst,
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def build_split(
    coco: COCO,
    selected_ids: list[int],
    source_split: str,
    subset_split: str,
    out_root: Path,
    cat_id_to_idx: dict[int, int],
    max_workers: int,
):
    img_out = out_root / "images" / subset_split
    lbl_out = out_root / "labels" / subset_split
    img_out.mkdir(parents=True, exist_ok=True)

    imgs = coco.loadImgs(selected_ids)

    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(download_image, img, source_split, img_out) for img in imgs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"download {subset_split}"):
            ok, path, msg = fut.result()
            if not ok:
                failures.append({"path": path, "error": msg})

    if failures:
        raise RuntimeError(f"{len(failures)} image downloads failed. First: {failures[0]}")

    ann_count = convert_labels(coco, selected_ids, lbl_out, cat_id_to_idx)
    return len(imgs), ann_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="/content/datasets/coco2017_subset_5k")
    ap.add_argument("--train-n", type=int, default=5000)
    ap.add_argument("--val-n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-train-per-class", type=int, default=30)
    ap.add_argument("--min-val-per-class", type=int, default=5)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--yaml-out", type=str, default="data/coco2017_subset_5k.yaml")
    ap.add_argument("--manifest-out", type=str, default="")
    args = ap.parse_args()

    out_root = Path(args.root)
    out_root.mkdir(parents=True, exist_ok=True)

    ann_dir = prepare_annotations(out_root)
    train_coco = COCO(str(ann_dir / "instances_train2017.json"))
    val_coco = COCO(str(ann_dir / "instances_val2017.json"))

    cats = sorted(train_coco.loadCats(train_coco.getCatIds()), key=lambda x: x["id"])
    names = [c["name"] for c in cats]
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}

    train_ids = select_images(train_coco, args.train_n, args.min_train_per_class, args.seed)
    val_ids = select_images(val_coco, args.val_n, args.min_val_per_class, args.seed + 1000)

    n_train, train_anns = build_split(train_coco, train_ids, "train2017", "train", out_root, cat_id_to_idx, args.workers)
    n_val, val_anns = build_split(val_coco, val_ids, "val2017", "val", out_root, cat_id_to_idx, args.workers)

    write_stats(train_coco, train_ids, cat_id_to_idx, out_root / "stats" / "train_class_stats.csv")
    write_stats(val_coco, val_ids, cat_id_to_idx, out_root / "stats" / "val_class_stats.csv")

    yaml_data = {
        "path": str(out_root),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }

    yaml_out = Path(args.yaml_out)
    yaml_out.parent.mkdir(parents=True, exist_ok=True)
    yaml_out.write_text(yaml.safe_dump(yaml_data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    manifest = {
        "root": str(out_root),
        "seed": args.seed,
        "train_n": n_train,
        "val_n": n_val,
        "train_annotations": train_anns,
        "val_annotations": val_anns,
        "nc": len(names),
        "names": names,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "yaml": str(yaml_out),
    }

    manifest_path = out_root / "subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.manifest_out:
        dst = Path(args.manifest_out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, dst)

    print(json.dumps({k: manifest[k] for k in ["root", "seed", "train_n", "val_n", "train_annotations", "val_annotations", "nc", "yaml"]}, indent=2))


if __name__ == "__main__":
    main()
