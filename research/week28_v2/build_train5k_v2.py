from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import urllib3
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm


def valid_ann(a):
    if a.get("iscrowd", 0):
        return False
    x, y, w, h = a["bbox"]
    return w > 1 and h > 1 and a.get("area", 0) > 1


def deterministic_select(coco, target, min_per_class, seed):
    rng = random.Random(seed)
    cats = sorted(coco.getCatIds())
    eligible, per = set(), {}
    for c in cats:
        ids = set()
        for ann in coco.loadAnns(coco.getAnnIds(catIds=[c], iscrowd=False)):
            if valid_ann(ann):
                ids.add(ann["image_id"])
                eligible.add(ann["image_id"])
        ids = sorted(ids)  # deterministic before shuffle
        rng.shuffle(ids)
        per[c] = ids

    out, used = [], set()
    for c in cats:
        added = 0
        for iid in per[c]:
            if iid not in used:
                out.append(iid)
                used.add(iid)
                added += 1
                if len(out) >= target:
                    return out
            if added >= min_per_class:
                break

    rem = sorted(eligible - used)
    rng.shuffle(rem)
    out.extend(rem[:target - len(out)])
    return out[:target]


def determine_tls_mode(url):
    try:
        r = requests.get(url, timeout=30, verify=True)
        r.raise_for_status()
        if len(r.content) < 1000:
            raise RuntimeError("probe response too small")
        print("COCO TLS probe PASS with verify=True")
        return True
    except Exception as first:
        print("Verified TLS probe failed:", repr(first))
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        r = requests.get(url, timeout=30, verify=False)
        r.raise_for_status()
        if len(r.content) < 1000:
            raise RuntimeError("verify=False probe response too small")
        print("COCO TLS fallback PASS with verify=False")
        return False


def download_image(img, dst, verify_tls, retries=5):
    p = dst / img["file_name"]
    if p.exists() and p.stat().st_size > 1000:
        return True, str(p), "exists"
    url = (img.get("coco_url") or
           f"https://images.cocodataset.org/train2017/{img['file_name']}").replace("http://", "https://")
    err = ""
    for i in range(retries):
        try:
            r = requests.get(url, timeout=45, verify=verify_tls)
            r.raise_for_status()
            p.write_bytes(r.content)
            if p.stat().st_size > 1000:
                return True, str(p), "downloaded"
        except Exception as e:
            err = repr(e)
            time.sleep(i + 1)
    return False, str(p), err


def label_fingerprint(labels):
    h = hashlib.sha256()
    for p in sorted(labels.glob("*.txt")):
        h.update(p.name.encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def validate_dataset(root, selected_ids, nc=80):
    root = Path(root)
    images = root / "images/train"
    labels = root / "labels/train"
    image_files = sorted(images.glob("*.jpg"))
    label_files = sorted(labels.glob("*.txt"))
    errors = []

    if len(image_files) != 5000:
        errors.append(f"image_count={len(image_files)}")
    if len(label_files) != 5000:
        errors.append(f"label_count={len(label_files)}")

    for p in tqdm(image_files, desc="verify-jpeg"):
        try:
            if p.stat().st_size <= 1000:
                raise RuntimeError("file too small")
            with Image.open(p) as im:
                im.verify()
        except Exception as e:
            errors.append(f"jpeg:{p.name}:{repr(e)}")

    literal_backslash_n = 0
    empty_labels = 0
    instances = 0
    class_counts = [0] * nc

    for p in label_files:
        raw = p.read_text(encoding="utf-8")
        if "\\n" in raw:
            literal_backslash_n += 1
            errors.append(f"literal_backslash_n:{p.name}")
            continue
        lines = [x.strip() for x in raw.splitlines() if x.strip()]
        if not lines:
            empty_labels += 1
            errors.append(f"empty_label:{p.name}")
            continue
        for lineno, line in enumerate(lines, 1):
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"tokens:{p.name}:{lineno}:{len(parts)}")
                continue
            try:
                cls = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
            except Exception as e:
                errors.append(f"parse:{p.name}:{lineno}:{repr(e)}")
                continue
            if not 0 <= cls < nc:
                errors.append(f"class:{p.name}:{lineno}:{cls}")
                continue
            vals = [cx, cy, bw, bh]
            if not all(math.isfinite(v) for v in vals):
                errors.append(f"nonfinite:{p.name}:{lineno}")
                continue
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                errors.append(f"range:{p.name}:{lineno}:{vals}")
                continue
            class_counts[cls] += 1
            instances += 1

    selected = set(map(int, selected_ids))
    image_ids = {int(p.stem) for p in image_files}
    label_ids = {int(p.stem) for p in label_files}
    if image_ids != selected:
        errors.append("selected_image_ids_mismatch")
    if label_ids != selected:
        errors.append("selected_label_ids_mismatch")

    return {
        "images": len(image_files),
        "labels": len(label_files),
        "instances": instances,
        "classes_with_instances": sum(x > 0 for x in class_counts),
        "min_class_instances": min(class_counts),
        "max_class_instances": max(class_counts),
        "literal_backslash_n_files": literal_backslash_n,
        "empty_label_files": empty_labels,
        "labels_sha256": label_fingerprint(labels),
        "errors": errors,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest-out", required=True)
    ap.add_argument("--selected-ids-json", default="")
    ap.add_argument("--selection-source", default="sorted_seed42_fallback")
    ap.add_argument("--target", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-per-class", type=int, default=30)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    root = Path(a.root)
    images = root / "images/train"
    labels = root / "labels/train"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for p in root.rglob("*.cache"):
        p.unlink()

    coco = COCO(a.ann)
    cats = sorted(coco.loadCats(coco.getCatIds()), key=lambda x: x["id"])
    cat2idx = {c["id"]: i for i, c in enumerate(cats)}

    if a.selected_ids_json:
        payload = json.loads(Path(a.selected_ids_json).read_text(encoding="utf-8"))
        selected = payload["selected_ids"] if isinstance(payload, dict) else payload
        selected = [int(x) for x in selected]
    else:
        selected = deterministic_select(coco, a.target, a.min_per_class, a.seed)

    if len(selected) != a.target or len(set(selected)) != a.target:
        raise RuntimeError("selected_ids must be 5000 unique IDs")

    imgs = coco.loadImgs(selected)
    probe_url = (imgs[0].get("coco_url") or
                 f"https://images.cocodataset.org/train2017/{imgs[0]['file_name']}").replace("http://", "https://")
    verify_tls = determine_tls_mode(probe_url)

    failures = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures = [ex.submit(download_image, im, images, verify_tls) for im in imgs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="train5k-images"):
            ok, p, msg = fut.result()
            if not ok:
                failures.append({"path": p, "error": msg})
    if failures:
        raise RuntimeError(f"download failures={len(failures)} first={failures[:3]}")

    # Regenerate all labels from COCO JSON. Never reuse v1 labels.
    for p in labels.glob("*.txt"):
        p.unlink()

    for iid in tqdm(selected, desc="write-labels"):
        im = coco.loadImgs([iid])[0]
        iw, ih = im["width"], im["height"]
        lines = []
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[iid], iscrowd=False)):
            if not valid_ann(ann):
                continue
            x, y, w, h = ann["bbox"]
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(iw), x + w), min(float(ih), y + h)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 1 or bh <= 1:
                continue
            cls = cat2idx[ann["category_id"]]
            cx = (x1 + x2) / 2 / iw
            cy = (y1 + y2) / 2 / ih
            lines.append(f"{cls} {cx:.8f} {cy:.8f} {bw/iw:.8f} {bh/ih:.8f}")

        # CRITICAL v2 FIX: real newline character.
        text = "\n".join(lines)
        if text:
            text += "\n"
        (labels / (Path(im["file_name"]).stem + ".txt")).write_text(text, encoding="utf-8")

    report = validate_dataset(root, selected)
    if report["errors"]:
        raise RuntimeError("Dataset validation FAIL:\n" + json.dumps(report["errors"][:30], indent=2))
    if report["classes_with_instances"] != 80:
        raise RuntimeError("All 80 classes must be present")

    manifest = {
        "version": "train5k_seed42_v1",
        "selection_source": a.selection_source,
        "seed": a.seed,
        "target": a.target,
        "min_per_class": a.min_per_class,
        **{k: v for k, v in report.items() if k != "errors"},
        "validation_error_count": 0,
        "selected_ids": selected,
    }
    Path(a.manifest_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.manifest_out).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (root / "canonical_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in manifest.items() if k != "selected_ids"}, indent=2))


if __name__ == "__main__":
    main()
