from pathlib import Path
import pandas as pd

def scan_repo(repo_root: Path, targets: dict, patterns: list[str]) -> pd.DataFrame:
    rows = []
    for label, file in targets.items():
        p = repo_root / file
        if not p.exists():
            rows.append([label, file, "-", "FILE_NOT_FOUND"])
            continue
        lines = p.read_text(errors="ignore").splitlines()
        for i, line in enumerate(lines, start=1):
            for pat in patterns:
                if pat in line:
                    rows.append([label, file, i, line.strip()[:220]])
    return pd.DataFrame(rows, columns=["module", "file", "line", "matched_code"])

if __name__ == "__main__":
    repo_root = Path(".").resolve()
    targets = {
        "YOLOv9 Head": "models/yolo.py",
        "YOLOv9 Dual Loss": "utils/loss_tal_dual.py",
        "YOLOv9 Train Dual": "train_dual.py",
        "YOLOv9 Val Dual": "val_dual.py",
        "YOLOv9 Detect Dual": "detect_dual.py",
    }
    patterns = [
        "class Detect",
        "class DDetect",
        "class DualDetect",
        "class DualDDetect",
        "class ComputeLoss",
        "TaskAlignedAssigner",
        "non_max_suppression",
        "compute_loss",
    ]
    df = scan_repo(repo_root, targets, patterns)
    out = repo_root / "research/week1/yolov9_code_structure.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df)
    print(f"saved: {out}")
