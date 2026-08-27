from pathlib import Path
import re
import yaml

Y = Path("models/yolo.py")
SRC = Path("models/detect/yolov9-s-v10dual-active-forward.yaml")
DST = Path("models/detect/yolov9-s-v10dual-training-v2.yaml")
text = Y.read_text(encoding="utf-8")
mark = "# WEEK28_V2_DUAL_TRAINING_HEAD"

if mark not in text:
    anchor = "class DualDSegment(DualDDetect):"
    if anchor not in text:
        raise RuntimeError("Safe insertion anchor not found")

    lines = [
        "# WEEK28_V2_DUAL_TRAINING_HEAD",
        "class V10DualTrainingDetectV2(V10DualActiveForwardDetect):",
        "    week28_dual_training = False",
        "",
        "    def forward(self, x):",
        "        if not self.training:",
        "            return V10DualActiveForwardDetect.forward(self, x)",
        "",
        "        # Model.__init__ stride bootstrap and all non-v2 training calls",
        "        # keep the native DDetect list-return contract.",
        "        if not self.week28_dual_training:",
        "            return DDetect.forward(self, x)",
        "",
        "        one2one_x = [xi.detach() for xi in x]",
        "        o2m_raw = DDetect.forward(self, x)",
        "        o2o_raw = []",
        "        for i in range(self.nl):",
        "            o2o_raw.append(torch.cat((",
        "                self.one2one_cv2[i](one2one_x[i]),",
        "                self.one2one_cv3[i](one2one_x[i])",
        "            ), 1))",
        "        return {'o2m': o2m_raw, 'o2o': o2o_raw}",
        "",
        "",
    ]
    text = text.replace(anchor, "\n".join(lines) + anchor, 1)

# Robustly add v2 class to the detection-head parse set that contains Week27 class.
matches = list(re.finditer(r"elif m in \{([^}]*)\}:", text, flags=re.DOTALL))
found = False
for m in matches:
    body = m.group(1)
    if "V10DualActiveForwardDetect" in body and "DualDDetect" in body:
        if "V10DualTrainingDetectV2" not in body:
            new_body = body.replace(
                "V10DualActiveForwardDetect",
                "V10DualActiveForwardDetect, V10DualTrainingDetectV2",
                1,
            )
            text = text[:m.start(1)] + new_body + text[m.end(1):]
        found = True
        break
if not found:
    raise RuntimeError("Week27 detection-head parse set not found")

positions = {
    "active": text.find("class V10DualActiveForwardDetect"),
    "dual_native": text.find("class DualDDetect(nn.Module):"),
    "week28v2": text.find("class V10DualTrainingDetectV2"),
    "segment": text.find("class DualDSegment(DualDDetect):"),
}
if not all(v >= 0 for v in positions.values()):
    raise RuntimeError(f"Class missing: {positions}")
if not (positions["active"] < positions["dual_native"] < positions["week28v2"] < positions["segment"]):
    raise RuntimeError(f"Unsafe class order: {positions}")

Y.write_text(text, encoding="utf-8")
cfg = yaml.safe_load(SRC.read_text(encoding="utf-8"))
cfg["head"][-1] = [[15, 18, 21], 1, "V10DualTrainingDetectV2", ["nc"]]
DST.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print("class order:", positions)
print("created:", DST)
