
from pathlib import Path
import yaml


Y = Path("models/yolo.py")
SRC = Path("models/detect/yolov9-s.yaml")
DST = Path(
    "models/detect/"
    "yolov9-s-v10o2m-equivalent.yaml"
)

text = Y.read_text(
    encoding="utf-8"
)

mark = "# WEEK25_V10_O2M_EQUIVALENCE"


# ============================================================
# 1. Add minimal Week25 O2M scaffold
# ============================================================

if mark not in text:

    anchor = "class DualDDetect(nn.Module):"

    if anchor not in text:
        raise RuntimeError(
            "DualDDetect anchor not found"
        )

    class_code = (
        "# WEEK25_V10_O2M_EQUIVALENCE\n"
        "class V10O2MDetect(DDetect):\n"
        "    # Week25 functional-equivalence scaffold.\n"
        "    # Exact official DDetect math/decode is reused.\n"
        "    # No O2O / no-NMS logic is introduced here.\n"
        "    pass\n"
        "\n"
        "\n"
    )

    text = text.replace(
        anchor,
        class_code + anchor,
        1,
    )


    # ========================================================
    # 2. Register V10O2MDetect in parse_model detection heads
    # ========================================================

    old = (
        "elif m in {Detect, DualDetect, TripleDetect, DDetect, "
        "DualDDetect, TripleDDetect, Segment, DSegment, "
        "DualDSegment, Panoptic}:"
    )

    new = (
        "elif m in {Detect, DualDetect, TripleDetect, DDetect, "
        "V10O2MDetect, DualDDetect, TripleDDetect, Segment, "
        "DSegment, DualDSegment, Panoptic}:"
    )

    if old not in text:
        raise RuntimeError(
            "parse_model detection-head set not found"
        )

    text = text.replace(
        old,
        new,
        1,
    )

    Y.write_text(
        text,
        encoding="utf-8",
    )

    print(
        "Patched models/yolo.py"
    )

else:

    print(
        "models/yolo.py already contains "
        "Week25 V10O2MDetect patch"
    )


# ============================================================
# 3. Create Week25 equivalence YAML
#
# Native main inference features:
# [15, 18, 21]
#
# Native semantic head:
# cv4 / cv5 / dfl2
#
# Target:
# V10O2MDetect cv2 / cv3 / dfl
# ============================================================

cfg = yaml.safe_load(
    SRC.read_text(
        encoding="utf-8"
    )
)

cfg["head"][-1] = [
    [15, 18, 21],
    1,
    "V10O2MDetect",
    ["nc"],
]

DST.write_text(
    yaml.safe_dump(
        cfg,
        sort_keys=False,
    ),
    encoding="utf-8",
)


print(
    "Created:",
    DST,
)

print(
    "Week25 V10O2M scaffold patch complete."
)
