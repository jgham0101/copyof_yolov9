
from pathlib import Path
import re

COPY_MARK = "# WEEK19_COPY_INIT_PATCH"
LOSS_MARK = "# WEEK19_O2M_WEIGHT_PATCH"


def patch_yolo(path: Path):
    text = path.read_text(encoding="utf-8")

    start = text.index("class V10DualDDetect")
    next_class = re.search(r"\nclass\s+\w+", text[start + 1:])
    end = len(text) if next_class is None else start + 1 + next_class.start()

    block = text[start:end]

    if COPY_MARK in block:
        print("already patched:", path, COPY_MARK)
        return

    # one2one head 정의가 모두 끝난 직후, DFL 정의 직전에 삽입
    needle = "        self.dfl = DFL(self.reg_max)\n"

    if needle not in block:
        raise RuntimeError(
            "Could not locate self.dfl = DFL(self.reg_max) "
            "inside V10DualDDetect."
        )

    insert = (
        '        self.copy_one2one_init = '
        '__import__("os").getenv("V10_COPY_INIT", "0") == "1"  '
        + COPY_MARK + '\n'
        '        if self.copy_one2one_init:\n'
        '            self.one2one_cv2.load_state_dict(self.cv2.state_dict(), strict=True)\n'
        '            self.one2one_cv3.load_state_dict(self.cv3.state_dict(), strict=True)\n'
        '\n'
    )

    block = block.replace(
        needle,
        insert + needle,
        1,
    )

    text = text[:start] + block + text[end:]
    path.write_text(text, encoding="utf-8")
    print("patched:", path, COPY_MARK)


def patch_loss(path: Path):
    text = path.read_text(encoding="utf-8")

    if LOSS_MARK in text:
        print("already patched:", path, LOSS_MARK)
        return

    # ComputeLoss 초기화 내부의 O2O top-k 설정 뒤에 실험용 weight 추가
    patterns = [
        r"(o2o_topk\s*=\s*int\(os\.getenv\([^\n]+\)\)\n)",
        r"(o2o_topk\s*=.*\n)",
    ]

    match = None
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            break

    if not match:
        raise RuntimeError("Could not locate o2o_topk initialization.")

    weight_line = (
        '        self.o2m_loss_weight = '
        'float(os.getenv("YOLO_O2M_LOSS_WEIGHT", "0.25"))  '
        + LOSS_MARK + '\n'
    )

    text = text[:match.end()] + weight_line + text[match.end():]

    # Week18 telemetry patch 기준 classification branch weighting
    replaced_cls = False

    cls_patterns = [
        (
            "        loss[1] = loss_cls_o2m\n"
            "        loss[1] *= 0.25\n"
            "        loss[1] += loss_cls_o2o\n",
            "        loss[1] = loss_cls_o2m * self.o2m_loss_weight\n"
            "        loss[1] += loss_cls_o2o\n",
        ),
        (
            "        loss[1] = self.BCEcls(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum # BCE\n"
            "        loss[1] *= 0.25\n",
            "        loss[1] = self.BCEcls(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum # BCE\n"
            "        loss[1] *= self.o2m_loss_weight\n",
        ),
    ]

    for old, new in cls_patterns:
        if old in text:
            text = text.replace(old, new, 1)
            replaced_cls = True
            break

    if not replaced_cls:
        raise RuntimeError("Could not patch O2M classification weighting.")

    # Week18 telemetry patch 기준 box/DFL weighting
    old_box = (
        "            loss[0] = loss_box_o2m * 0.25\n"
        "            loss[2] = loss_dfl_o2m * 0.25\n"
    )
    new_box = (
        "            loss[0] = loss_box_o2m * self.o2m_loss_weight\n"
        "            loss[2] = loss_dfl_o2m * self.o2m_loss_weight\n"
    )

    if old_box in text:
        text = text.replace(old_box, new_box, 1)
    else:
        # older layout fallback
        old_legacy = (
            "            loss[0] *= 0.25\n"
            "            loss[2] *= 0.25\n"
        )
        new_legacy = (
            "            loss[0] *= self.o2m_loss_weight\n"
            "            loss[2] *= self.o2m_loss_weight\n"
        )
        if old_legacy in text:
            text = text.replace(old_legacy, new_legacy, 1)
        else:
            raise RuntimeError("Could not patch O2M box/DFL weighting.")

    path.write_text(text, encoding="utf-8")
    print("patched:", path, LOSS_MARK)


def main():
    patch_yolo(Path("models/yolo.py"))
    patch_loss(Path("utils/loss_tal_dual.py"))


if __name__ == "__main__":
    main()
