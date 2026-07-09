
import argparse
from pathlib import Path

PATCH_MARK = "# WEEK18_TELEMETRY_PATCH"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="utils/loss_tal_dual.py")
    ap.add_argument("--backup", default="")
    args = ap.parse_args()

    p = Path(args.file)
    assert p.exists(), p
    text = p.read_text(encoding="utf-8")

    if PATCH_MARK in text:
        print("already patched:", p)
        return

    backup = Path(args.backup) if args.backup else p.with_suffix(p.suffix + ".week18_backup")
    backup.write_text(text, encoding="utf-8")
    print("backup:", backup)

    old_cls = """        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores) / target_scores_sum  # VFL way
        loss[1] = self.BCEcls(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum # BCE
        loss[1] *= 0.25
        loss[1] += self.BCEcls(pred_scores2, target_scores2.to(dtype)).sum() / target_scores_sum2 # BCE
"""
    if old_cls not in text:
        old_cls = """        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.BCEcls(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum # BCE
        loss[1] *= 0.25
        loss[1] += self.BCEcls(pred_scores2, target_scores2.to(dtype)).sum() / target_scores_sum2 # BCE
"""

    new_cls = f"""        # cls loss
        # {PATCH_MARK}: keep original math but expose separate branch components.
        loss_cls_o2m = self.BCEcls(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        loss_cls_o2o = self.BCEcls(pred_scores2, target_scores2.to(dtype)).sum() / target_scores_sum2  # BCE
        loss[1] = loss_cls_o2m
        loss[1] *= 0.25
        loss[1] += loss_cls_o2o
"""

    assert old_cls in text, "Could not find cls loss block"
    text = text.replace(old_cls, new_cls, 1)

    old_bbox = """        # bbox loss
        if fg_mask.sum():
            loss[0], loss[2], iou = self.bbox_loss(pred_distri,
                                                   pred_bboxes,
                                                   anchor_points,
                                                   target_bboxes,
                                                   target_scores,
                                                   target_scores_sum,
                                                   fg_mask)
            loss[0] *= 0.25
            loss[2] *= 0.25
        if fg_mask2.sum():
            loss0_, loss2_, iou2 = self.bbox_loss2(pred_distri2,
                                                   pred_bboxes2,
                                                   anchor_points,
                                                   target_bboxes2,
                                                   target_scores2,
                                                   target_scores_sum2,
                                                   fg_mask2)
            loss[0] += loss0_
            loss[2] += loss2_

        loss[0] *= 7.5  # box gain
"""
    new_bbox = f"""        # bbox loss
        # {PATCH_MARK}: keep original math but expose separate branch components.
        loss_box_o2m = pred_scores.new_tensor(0.0)
        loss_dfl_o2m = pred_scores.new_tensor(0.0)
        loss_box_o2o = pred_scores.new_tensor(0.0)
        loss_dfl_o2o = pred_scores.new_tensor(0.0)
        if fg_mask.sum():
            loss_box_o2m, loss_dfl_o2m, iou = self.bbox_loss(pred_distri,
                                                             pred_bboxes,
                                                             anchor_points,
                                                             target_bboxes,
                                                             target_scores,
                                                             target_scores_sum,
                                                             fg_mask)
            loss[0] = loss_box_o2m * 0.25
            loss[2] = loss_dfl_o2m * 0.25
        if fg_mask2.sum():
            loss_box_o2o, loss_dfl_o2o, iou2 = self.bbox_loss2(pred_distri2,
                                                               pred_bboxes2,
                                                               anchor_points,
                                                               target_bboxes2,
                                                               target_scores2,
                                                               target_scores_sum2,
                                                               fg_mask2)
            loss[0] += loss_box_o2o
            loss[2] += loss_dfl_o2o

        # {PATCH_MARK}: JSONL telemetry for assignment/loss diagnosis. Disabled unless YOLO_LOSS_TELEMETRY is set.
        _telemetry_path = os.getenv("YOLO_LOSS_TELEMETRY", "")
        if _telemetry_path:
            try:
                import json as _json
                if not hasattr(self, "_week18_telemetry_step"):
                    self._week18_telemetry_step = 0
                self._week18_telemetry_step += 1
                _every = int(os.getenv("YOLO_LOSS_TELEMETRY_EVERY", "1"))
                if _every <= 0:
                    _every = 1

                def _tf(v):
                    try:
                        return float(v.detach().float().cpu().item()) if torch.is_tensor(v) else float(v)
                    except Exception:
                        return None

                def _ti(v):
                    try:
                        return int(v.detach().cpu().item()) if torch.is_tensor(v) else int(v)
                    except Exception:
                        return None

                if self._week18_telemetry_step % _every == 0:
                    _row = {{
                        "step": int(self._week18_telemetry_step),
                        "batch_size": int(batch_size),
                        "num_targets": int(targets.shape[0]) if hasattr(targets, "shape") else None,
                        "fg_mask_sum_o2m": _ti(fg_mask.sum()),
                        "fg_mask_sum_o2o": _ti(fg_mask2.sum()),
                        "target_scores_sum_o2m": _tf(target_scores_sum),
                        "target_scores_sum_o2o": _tf(target_scores_sum2),
                        "loss_cls_o2m_raw": _tf(loss_cls_o2m),
                        "loss_cls_o2o_raw": _tf(loss_cls_o2o),
                        "loss_box_o2m_raw": _tf(loss_box_o2m),
                        "loss_box_o2o_raw": _tf(loss_box_o2o),
                        "loss_dfl_o2m_raw": _tf(loss_dfl_o2m),
                        "loss_dfl_o2o_raw": _tf(loss_dfl_o2o),
                        "loss_box_combined_pre_gain": _tf(loss[0]),
                        "loss_cls_combined_pre_gain": _tf(loss[1]),
                        "loss_dfl_combined_pre_gain": _tf(loss[2]),
                    }}
                    with open(_telemetry_path, "a", encoding="utf-8") as _f:
                        _f.write(_json.dumps(_row) + "\\\\n")
            except Exception as _e:
                if os.getenv("YOLO_LOSS_TELEMETRY_STRICT", "0") == "1":
                    raise

        loss[0] *= 7.5  # box gain
"""

    assert old_bbox in text, "Could not find bbox loss block"
    text = text.replace(old_bbox, new_bbox, 1)

    p.write_text(text, encoding="utf-8")
    print("patched:", p)

if __name__ == "__main__":
    main()
