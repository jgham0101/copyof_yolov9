
from __future__ import annotations

import argparse
import json
import random
import time
import warnings

from pathlib import Path
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

import requests
import urllib3
import yaml

from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO


# ------------------------------------------------------------
# Annotation helpers
# ------------------------------------------------------------

def valid(ann):

    if ann.get("iscrowd", 0):
        return False

    x, y, w, h = ann["bbox"]

    return (
        w > 1
        and h > 1
        and ann.get("area", 0) > 1
    )


def select(
    coco,
    target,
    min_per_class,
    seed,
):

    rng = random.Random(seed)

    cats = sorted(
        coco.getCatIds()
    )

    eligible = set()
    per_class = {}

    # --------------------------------------------------------
    # Collect images containing each category
    # --------------------------------------------------------

    for cat_id in cats:

        ids = set()

        anns = coco.loadAnns(
            coco.getAnnIds(
                catIds=[cat_id],
                iscrowd=False,
            )
        )

        for ann in anns:

            if valid(ann):

                ids.add(
                    ann["image_id"]
                )

                eligible.add(
                    ann["image_id"]
                )

        ids = list(ids)

        rng.shuffle(ids)

        per_class[cat_id] = ids

    # --------------------------------------------------------
    # First ensure broad class coverage
    # --------------------------------------------------------

    selected = []
    selected_set = set()

    for cat_id in cats:

        added = 0

        for image_id in per_class[cat_id]:

            if image_id not in selected_set:

                selected.append(
                    image_id
                )

                selected_set.add(
                    image_id
                )

                added += 1

                if len(selected) >= target:
                    return selected

            if added >= min_per_class:
                break

    # --------------------------------------------------------
    # Fill remaining slots randomly
    # --------------------------------------------------------

    remaining = list(
        eligible - selected_set
    )

    rng.shuffle(
        remaining
    )

    selected.extend(
        remaining[
            : max(
                0,
                target - len(selected),
            )
        ]
    )

    return selected[:target]


# ------------------------------------------------------------
# Download helpers
# ------------------------------------------------------------

def image_is_valid(
    path: Path,
    expected_width=None,
    expected_height=None,
):

    if not path.exists():
        return False

    if path.stat().st_size < 1000:
        return False

    try:

        with Image.open(path) as im:

            width, height = im.size

            im.verify()

        if (
            expected_width is not None
            and width != expected_width
        ):
            return False

        if (
            expected_height is not None
            and height != expected_height
        ):
            return False

        return True

    except Exception:

        return False


def save_response(
    response,
    dst,
    expected_width,
    expected_height,
):

    tmp = dst.with_suffix(
        dst.suffix + ".part"
    )

    tmp.write_bytes(
        response.content
    )

    if image_is_valid(
        tmp,
        expected_width,
        expected_height,
    ):

        tmp.replace(dst)

        return True

    if tmp.exists():
        tmp.unlink()

    return False


def try_request(
    session,
    url,
    dst,
    expected_width,
    expected_height,
    verify=True,
    timeout=30,
):

    try:

        response = session.get(
            url,
            timeout=timeout,
            verify=verify,
        )

        response.raise_for_status()

        return save_response(
            response,
            dst,
            expected_width,
            expected_height,
        ), None

    except Exception as e:

        return False, repr(e)


def download(
    img,
    dst_dir,
    retries=2,
):

    dst = (
        dst_dir
        / img["file_name"]
    )

    expected_width = int(
        img["width"]
    )

    expected_height = int(
        img["height"]
    )

    # --------------------------------------------------------
    # Existing valid file -> reuse
    # --------------------------------------------------------

    if image_is_valid(
        dst,
        expected_width,
        expected_height,
    ):

        return (
            True,
            str(dst),
            "exists",
        )

    if dst.exists():
        dst.unlink()

    # --------------------------------------------------------
    # Candidate URLs
    #
    # 1. Original Flickr URL from COCO annotation
    # 2. Official COCO HTTPS
    #
    # COCO HTTPS with verify=False is used only as
    # a final fallback because this Colab runtime has shown
    # certificate hostname verification failures.
    # --------------------------------------------------------

    candidates = []

    flickr = img.get(
        "flickr_url"
    )

    if flickr:

        flickr = flickr.replace(
            "http://",
            "https://",
        )

        candidates.append(
            (
                "flickr",
                flickr,
                True,
            )
        )

    coco_url = img.get(
        "coco_url"
    )

    if not coco_url:

        coco_url = (
            "https://images.cocodataset.org/"
            "train2017/"
            + img["file_name"]
        )

    coco_url = coco_url.replace(
        "http://",
        "https://",
    )

    candidates.append(
        (
            "coco_https",
            coco_url,
            True,
        )
    )

    candidates.append(
        (
            "coco_ssl_fallback",
            coco_url,
            False,
        )
    )

    session = requests.Session()

    session.headers.update({
        "User-Agent":
            "Mozilla/5.0 Week24-COCO-Downloader"
    })

    errors = []

    # --------------------------------------------------------
    # Try candidates
    # --------------------------------------------------------

    for (
        source,
        url,
        verify_ssl,
    ) in candidates:

        for attempt in range(
            retries
        ):

            ok, err = try_request(
                session=session,
                url=url,
                dst=dst,
                expected_width=expected_width,
                expected_height=expected_height,
                verify=verify_ssl,
            )

            if ok:

                return (
                    True,
                    str(dst),
                    source,
                )

            errors.append({
                "source": source,
                "attempt": attempt + 1,
                "error": err,
            })

            time.sleep(
                0.3 * (attempt + 1)
            )

    return (
        False,
        str(dst),
        errors,
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--ann",
        required=True,
    )

    ap.add_argument(
        "--root",
        required=True,
    )

    ap.add_argument(
        "--target",
        type=int,
        default=5000,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    ap.add_argument(
        "--min-per-class",
        type=int,
        default=30,
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--repo-coco-yaml",
        required=True,
    )

    ap.add_argument(
        "--yaml-out",
        required=True,
    )

    ap.add_argument(
        "--manifest-out",
        required=True,
    )

    args = ap.parse_args()

    # suppress warnings only for the explicitly validated
    # final COCO SSL fallback.
    urllib3.disable_warnings(
        urllib3.exceptions.InsecureRequestWarning
    )

    root = Path(
        args.root
    )

    img_dir = (
        root
        / "images"
        / "train"
    )

    label_dir = (
        root
        / "labels"
        / "train"
    )

    img_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load COCO annotations
    # --------------------------------------------------------

    coco = COCO(
        args.ann
    )

    cats = sorted(
        coco.loadCats(
            coco.getCatIds()
        ),
        key=lambda x: x["id"],
    )

    cat_to_idx = {
        cat["id"]: i
        for i, cat
        in enumerate(cats)
    }

    # --------------------------------------------------------
    # Deterministic 5k selection
    # --------------------------------------------------------

    selected = select(
        coco,
        args.target,
        args.min_per_class,
        args.seed,
    )

    assert len(selected) == args.target, (
        len(selected)
    )

    images = coco.loadImgs(
        selected
    )

    print(
        "Selected images:",
        len(images),
    )

    # --------------------------------------------------------
    # Download only selected 5,000 images
    # --------------------------------------------------------

    failures = []

    source_counts = {
        "exists": 0,
        "flickr": 0,
        "coco_https": 0,
        "coco_ssl_fallback": 0,
    }

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:

        futures = [
            executor.submit(
                download,
                img,
                img_dir,
            )
            for img in images
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="train5k",
        ):

            ok, path, result = (
                future.result()
            )

            if ok:

                source_counts[
                    result
                ] = (
                    source_counts.get(
                        result,
                        0,
                    )
                    + 1
                )

            else:

                failures.append({
                    "path": path,
                    "errors": result,
                })

    print(
        "\nDownload source counts:"
    )

    print(
        json.dumps(
            source_counts,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # Require all 5,000
    # --------------------------------------------------------

    if failures:

        failure_path = (
            Path(args.manifest_out)
            .with_name(
                "week24_train5k_download_failures.json"
            )
        )

        failure_path.write_text(
            json.dumps(
                failures,
                indent=2,
            ),
            encoding="utf-8",
        )

        raise RuntimeError(
            f"download failures={len(failures)}; "
            f"details={failure_path}; "
            f"first={failures[:2]}"
        )

    downloaded = list(
        img_dir.glob("*.jpg")
    )

    assert len(downloaded) == args.target, (
        f"Expected {args.target} images, "
        f"found {len(downloaded)}"
    )

    # --------------------------------------------------------
    # Generate YOLO labels
    # --------------------------------------------------------

    class_counts = [
        0
    ] * len(cats)

    instances = 0

    for image_id in selected:

        img = coco.loadImgs(
            [image_id]
        )[0]

        iw = img["width"]
        ih = img["height"]

        lines = []

        anns = coco.loadAnns(
            coco.getAnnIds(
                imgIds=[image_id],
                iscrowd=False,
            )
        )

        for ann in anns:

            if not valid(ann):
                continue

            x, y, w, h = (
                ann["bbox"]
            )

            x1 = max(
                0.0,
                x,
            )

            y1 = max(
                0.0,
                y,
            )

            x2 = min(
                float(iw),
                x + w,
            )

            y2 = min(
                float(ih),
                y + h,
            )

            bw = x2 - x1
            bh = y2 - y1

            if (
                bw <= 1
                or bh <= 1
            ):
                continue

            cls = cat_to_idx[
                ann["category_id"]
            ]

            class_counts[
                cls
            ] += 1

            instances += 1

            cx = (
                (x1 + x2)
                / 2.0
                / iw
            )

            cy = (
                (y1 + y2)
                / 2.0
                / ih
            )

            nw = (
                bw / iw
            )

            nh = (
                bh / ih
            )

            lines.append(
                f"{cls} "
                f"{cx:.8f} "
                f"{cy:.8f} "
                f"{nw:.8f} "
                f"{nh:.8f}"
            )

        label_path = (
            label_dir
            / (
                Path(
                    img["file_name"]
                ).stem
                + ".txt"
            )
        )

        label_path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

    # --------------------------------------------------------
    # Dataset YAML
    # --------------------------------------------------------

    base_yaml = yaml.safe_load(
        Path(
            args.repo_coco_yaml
        ).read_text(
            encoding="utf-8"
        )
    )

    dataset_yaml = {
        "path": "/",
        "train": str(
            img_dir
        ),
        "val":
            "/content/datasets/coco/"
            "images/val2017",
        "names":
            base_yaml["names"],
    }

    yaml_out = Path(
        args.yaml_out
    )

    yaml_out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    yaml_out.write_text(
        yaml.safe_dump(
            dataset_yaml,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {
        "seed":
            args.seed,

        "num_images":
            len(
                list(
                    img_dir.glob(
                        "*.jpg"
                    )
                )
            ),

        "num_labels":
            len(
                list(
                    label_dir.glob(
                        "*.txt"
                    )
                )
            ),

        "num_instances":
            instances,

        "classes_with_instances":
            sum(
                x > 0
                for x in class_counts
            ),

        "min_class_instances":
            min(
                class_counts
            ),

        "max_class_instances":
            max(
                class_counts
            ),

        "download_source_counts":
            source_counts,

        "selected_ids":
            selected,
    }

    manifest_out = Path(
        args.manifest_out
    )

    manifest_out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_out.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== TRAIN5K COMPLETE ==="
    )

    print(
        json.dumps(
            {
                key: value
                for key, value
                in manifest.items()
                if key != "selected_ids"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
