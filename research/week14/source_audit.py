
import argparse, csv, json, re
from pathlib import Path

KEYWORDS = [
    "one2one", "one-to-one", "one2many", "one-to-many", "detach",
    "assigner", "TaskAlignedAssigner", "tal", "loss", "bbox_loss",
    "cls_loss", "dfl_loss", "topk", "top_k", "select_one2one",
    "V10DualDDetect"
]

def context(lines, idx, radius=5):
    lo, hi = max(0, idx - radius), min(len(lines), idx + radius + 1)
    return "".join(f"{i+1}: {lines[i]}" for i in range(lo, hi))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    repo = Path(args.repo)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for pattern in ["models/**/*.py", "utils/**/*.py", "train*.py", "val*.py", "research/**/*.py"]:
        files.extend(repo.glob(pattern))
    files = sorted(set(files))

    rows, contexts = [], []
    important = {"detach", "one2one", "one2many", "assigner", "taskalignedassigner", "v10dualdetect"}

    for path in files:
        if "__pycache__" in str(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        rel = path.relative_to(repo)
        lines = text.splitlines(keepends=True)
        for i, line in enumerate(lines):
            low = line.lower()
            for kw in KEYWORDS:
                if kw.lower() in low:
                    rows.append({"file": str(rel), "line": i + 1, "keyword": kw, "text": line.strip()})
                    if kw.lower() in important:
                        contexts.append({"file": str(rel), "line": i + 1, "keyword": kw, "context": context(lines, i)})

    with (out_dir / "source_keyword_hits.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "line", "keyword", "text"])
        w.writeheader()
        w.writerows(rows)

    md = ["# Week 14 Source Audit", ""]
    for c in contexts:
        md += [f"## {c['file']}:{c['line']} — {c['keyword']}", "```python", c["context"].rstrip(), "```", ""]
    (out_dir / "source_contexts.md").write_text("\n".join(md), encoding="utf-8")

    summary = {
        "num_files_scanned": len(files),
        "num_keyword_hits": len(rows),
        "num_contexts": len(contexts),
        "keywords": KEYWORDS,
    }
    (out_dir / "source_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
