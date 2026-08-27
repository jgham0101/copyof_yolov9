from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def load_builder(path):
    spec = importlib.util.spec_from_file_location("week28v2_builder", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--builder", required=True)
    ap.add_argument("--val-images", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    mod = load_builder(a.builder)
    report = mod.validate_dataset(a.root, manifest["selected_ids"])

    train_names = {p.name for p in (Path(a.root) / "images/train").glob("*.jpg")}
    val_names = {p.name for p in Path(a.val_images).glob("*.jpg")}
    overlap = len(train_names & val_names)

    out = {
        **{k:v for k,v in report.items() if k != "errors"},
        "error_count": len(report["errors"]),
        "manifest_labels_sha256": manifest["labels_sha256"],
        "labels_fingerprint_match": report["labels_sha256"] == manifest["labels_sha256"],
        "train_val_filename_overlap": overlap,
    }
    out["pass"] = bool(
        out["images"] == 5000
        and out["labels"] == 5000
        and out["error_count"] == 0
        and out["literal_backslash_n_files"] == 0
        and out["empty_label_files"] == 0
        and out["classes_with_instances"] == 80
        and out["labels_fingerprint_match"]
        and overlap == 0
    )

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    if not out["pass"]:
        raise RuntimeError("Week28 v2 train5k integrity gate FAIL")


if __name__ == "__main__":
    main()
