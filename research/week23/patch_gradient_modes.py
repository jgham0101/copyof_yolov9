
import argparse
from pathlib import Path

MARK = "# WEEK23_GRADIENT_MODE_PATCH"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="utils/loss_tal_dual.py")
    ap.add_argument("--backup", default="")
    args = ap.parse_args()

    path = Path(args.file)
    assert path.exists(), path

    text = path.read_text(encoding="utf-8")

    if MARK in text:
        print("already patched:", path)
        return

    assert "# WEEK22_REP_RANK_PATCH" in text, (
        "Week22 representative ranking patch not found."
    )

    backup = (
        Path(args.backup)
        if args.backup
        else path.with_suffix(path.suffix + ".week23_backup")
    )
    backup.write_text(text, encoding="utf-8")

    old_cfg = (
        '        self.rep_rank_iou_threshold = '
        'float(os.getenv("YOLO_REP_RANK_IOU", "0.5"))\n'
    )

    new_cfg = (
        old_cfg
        + '        self.rep_rank_mode = '
          'os.getenv("YOLO_REP_RANK_MODE", "pairwise").strip().lower()  '
          + MARK + '\n'
        + '        if self.rep_rank_mode not in '
          '{"pairwise", "positive_only", "negative_only"}:\n'
        + '            raise ValueError('
          'f"Unsupported YOLO_REP_RANK_MODE={self.rep_rank_mode}")\n'
    )

    if old_cfg not in text:
        raise RuntimeError(
            "rep_rank_iou_threshold config anchor not found"
        )

    text = text.replace(old_cfg, new_cfg, 1)

    old_loss = (
        "                pair_loss = torch.nn.functional.softplus(\n"
        "                    neg_logit\n"
        "                    - pos_logit\n"
        "                )\n"
    )

    new_loss = (
        "                # WEEK23_GRADIENT_MODE_PATCH:\n"
        "                # Same mined pair; only gradient destination changes.\n"
        "                if self.rep_rank_mode == \"positive_only\":\n"
        "                    pos_for_loss = pos_logit\n"
        "                    neg_for_loss = neg_logit.detach()\n"
        "\n"
        "                elif self.rep_rank_mode == \"negative_only\":\n"
        "                    pos_for_loss = pos_logit.detach()\n"
        "                    neg_for_loss = neg_logit\n"
        "\n"
        "                else:  # pairwise\n"
        "                    pos_for_loss = pos_logit\n"
        "                    neg_for_loss = neg_logit\n"
        "\n"
        "                pair_loss = torch.nn.functional.softplus(\n"
        "                    neg_for_loss\n"
        "                    - pos_for_loss\n"
        "                )\n"
    )

    if old_loss not in text:
        raise RuntimeError("Week22 pair_loss block not found")

    text = text.replace(old_loss, new_loss, 1)
    path.write_text(text, encoding="utf-8")

    print("patched:", path)
    print("backup :", backup)


if __name__ == "__main__":
    main()
