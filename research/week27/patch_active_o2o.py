
from pathlib import Path
import yaml


Y = Path("models/yolo.py")

SRC = Path(
    "models/detect/"
    "yolov9-s-v10dual-dormant.yaml"
)

DST = Path(
    "models/detect/"
    "yolov9-s-v10dual-active-forward.yaml"
)


text = Y.read_text(
    encoding="utf-8"
)

mark = "# WEEK27_ACTIVE_O2O_FORWARD"


# ============================================================
# 0. Repair the malformed Week27 line if the previous patch
#    already inserted escaped triple quotes into models/yolo.py
# ============================================================

malformed_docstring = (
    r'    \"\"\"Week27: execute detached O2O forward, '
    r'but keep O2M as inference output.\"\"\"'
)

fixed_comment = (
    "    # Week27: execute detached O2O forward, "
    "but keep O2M as inference output."
)

if malformed_docstring in text:

    text = text.replace(
        malformed_docstring,
        fixed_comment,
        1,
    )

    print(
        "Repaired malformed escaped docstring "
        "from previous Cell 7 execution."
    )


# ============================================================
# 1. Add Week27 class if it does not exist yet
# ============================================================

if mark not in text:

    anchor = "class DualDDetect(nn.Module):"

    if anchor not in text:
        raise RuntimeError(
            "DualDDetect anchor not found"
        )


    class_code = (
        "# WEEK27_ACTIVE_O2O_FORWARD\n"
        "class V10DualActiveForwardDetect("
        "V10DualDormantDetect):\n"
        "    # Week27: execute detached O2O forward, "
        "but keep O2M as inference output.\n"
        "\n"
        "    def forward(self, x):\n"
        "        # Preserve original feature values before "
        "DDetect replaces x[i] with prediction tensors.\n"
        "        one2one_x = [xi.detach() for xi in x]\n"
        "\n"
        "        # Validated Week25/26 O2M path.\n"
        "        # This remains the externally returned "
        "inference path.\n"
        "        o2m_out = DDetect.forward(self, x)\n"
        "\n"
        "        # ------------------------------------------------\n"
        "        # Active O2O forward\n"
        "        # ------------------------------------------------\n"
        "        o2o_raw = []\n"
        "\n"
        "        for i in range(self.nl):\n"
        "            o2o_raw.append(\n"
        "                torch.cat(\n"
        "                    (\n"
        "                        self.one2one_cv2[i](\n"
        "                            one2one_x[i]\n"
        "                        ),\n"
        "                        self.one2one_cv3[i](\n"
        "                            one2one_x[i]\n"
        "                        ),\n"
        "                    ),\n"
        "                    1,\n"
        "                )\n"
        "            )\n"
        "\n"
        "        audit_return = (\n"
        "            os.getenv(\n"
        "                'YOLO_WEEK27_AUDIT_RETURN_O2O',\n"
        "                '0',\n"
        "            )\n"
        "            == '1'\n"
        "        )\n"
        "\n"
        "        # During training-mode diagnostic forward,\n"
        "        # DDetect.forward returns raw O2M tensors.\n"
        "        if self.training:\n"
        "\n"
        "            if audit_return:\n"
        "                return {\n"
        "                    'o2m': o2m_out,\n"
        "                    'o2o_raw': o2o_raw,\n"
        "                    'o2o_decoded': None,\n"
        "                }\n"
        "\n"
        "            return o2m_out\n"
        "\n"
        "        # ------------------------------------------------\n"
        "        # O2O decode\n"
        "        # Uses the SAME anchors / strides / DFL rule as O2M\n"
        "        # ------------------------------------------------\n"
        "        shape = one2one_x[0].shape\n"
        "\n"
        "        box2, cls2 = torch.cat(\n"
        "            [\n"
        "                di.view(\n"
        "                    shape[0],\n"
        "                    self.no,\n"
        "                    -1,\n"
        "                )\n"
        "                for di in o2o_raw\n"
        "            ],\n"
        "            2,\n"
        "        ).split(\n"
        "            (\n"
        "                self.reg_max * 4,\n"
        "                self.nc,\n"
        "            ),\n"
        "            1,\n"
        "        )\n"
        "\n"
        "        dbox2 = dist2bbox(\n"
        "            self.dfl(box2),\n"
        "            self.anchors.unsqueeze(0),\n"
        "            xywh=True,\n"
        "            dim=1,\n"
        "        ) * self.strides\n"
        "\n"
        "        o2o_decoded = torch.cat(\n"
        "            (\n"
        "                dbox2,\n"
        "                cls2.sigmoid(),\n"
        "            ),\n"
        "            1,\n"
        "        )\n"
        "\n"
        "        # Audit mode exposes both branches.\n"
        "        if audit_return:\n"
        "            return {\n"
        "                'o2m': o2m_out,\n"
        "                'o2o_raw': o2o_raw,\n"
        "                'o2o_decoded': o2o_decoded,\n"
        "            }\n"
        "\n"
        "        # Normal val.py interface remains exactly the\n"
        "        # validated O2M DDetect output.\n"
        "        return o2m_out\n"
        "\n"
        "\n"
    )


    text = text.replace(
        anchor,
        class_code + anchor,
        1,
    )

    print(
        "Inserted V10DualActiveForwardDetect."
    )

else:

    print(
        "Week27 class already exists; "
        "only repairing/verifying it."
    )


# ============================================================
# 2. Ensure parse_model recognizes the Week27 head
# ============================================================

old_parse = (
    "elif m in {Detect, DualDetect, TripleDetect, DDetect, "
    "V10O2MDetect, V10DualDormantDetect, DualDDetect, "
    "TripleDDetect, Segment, DSegment, DualDSegment, Panoptic}:"
)

new_parse = (
    "elif m in {Detect, DualDetect, TripleDetect, DDetect, "
    "V10O2MDetect, V10DualDormantDetect, "
    "V10DualActiveForwardDetect, DualDDetect, "
    "TripleDDetect, Segment, DSegment, "
    "DualDSegment, Panoptic}:"
)

if old_parse in text:

    text = text.replace(
        old_parse,
        new_parse,
        1,
    )


# ============================================================
# 3. Final source sanity
# ============================================================

# Escaped triple quotes must never remain in generated source.
if r'\"\"\"Week27:' in text:
    raise RuntimeError(
        "Malformed escaped Week27 docstring still exists."
    )

if mark not in text:
    raise RuntimeError(
        "Week27 class marker missing after patch."
    )

if "V10DualActiveForwardDetect" not in text:
    raise RuntimeError(
        "Week27 class missing after patch."
    )


Y.write_text(
    text,
    encoding="utf-8",
)


# ============================================================
# 4. Create active-forward YAML
# ============================================================

if not SRC.exists():
    raise FileNotFoundError(
        f"Week26 YAML not found: {SRC}"
    )

cfg = yaml.safe_load(
    SRC.read_text(
        encoding="utf-8"
    )
)

cfg["head"][-1] = [
    [15, 18, 21],
    1,
    "V10DualActiveForwardDetect",
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
    "patched:",
    Y,
)

print(
    "created:",
    DST,
)

print(
    "Week27 active O2O patch complete."
)
