
import argparse, copy, json, sys
from pathlib import Path
import torch, yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.yolo import Model
from utils.general import intersect_dicts


def torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def get_model_from_ckpt(ckpt):
    if isinstance(ckpt, dict):
        return ckpt.get('ema') or ckpt.get('model')
    return ckpt


def copy_branch_weights(model):
    head = model.model[-1]
    report = {'head_class': head.__class__.__name__, 'copied': [], 'errors': []}
    pairs = [('cv2', 'one2one_cv2'), ('cv3', 'one2one_cv3'), ('dfl', 'dfl2')]
    for src_name, dst_name in pairs:
        if not hasattr(head, src_name) or not hasattr(head, dst_name):
            report['errors'].append(f'missing {src_name} or {dst_name}')
            continue
        src = getattr(head, src_name)
        dst = getattr(head, dst_name)
        try:
            if isinstance(src, torch.nn.ModuleList) and isinstance(dst, torch.nn.ModuleList):
                assert len(src) == len(dst), f'{src_name}/{dst_name} length mismatch'
                for i, (s, d) in enumerate(zip(src, dst)):
                    d.load_state_dict(copy.deepcopy(s.state_dict()), strict=True)
                    report['copied'].append(f'{src_name}[{i}] -> {dst_name}[{i}]')
            else:
                dst.load_state_dict(copy.deepcopy(src.state_dict()), strict=True)
                report['copied'].append(f'{src_name} -> {dst_name}')
        except Exception as e:
            report['errors'].append(f'{src_name}->{dst_name}: {repr(e)}')
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--source-weights', required=True)
    ap.add_argument('--out-weights', required=True)
    ap.add_argument('--out-report', required=True)
    args = ap.parse_args()

    data = yaml.safe_load(Path(args.data).read_text(encoding='utf-8'))
    nc = int(data.get('nc', 80))

    model = Model(args.cfg, ch=3, nc=nc, anchors=None)
    model.float()

    ckpt = torch_load(args.source_weights)
    src_model = get_model_from_ckpt(ckpt)
    assert src_model is not None, 'source checkpoint has no model/ema'

    src_state = src_model.float().state_dict()
    dst_state = model.state_dict()
    compatible = intersect_dicts(src_state, dst_state, exclude=[])
    model.load_state_dict(compatible, strict=False)

    branch_report = copy_branch_weights(model)

    out = Path(args.out_weights)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_out = {
        'epoch': -1,
        'best_fitness': None,
        'model': copy.deepcopy(model).half(),
        'ema': None,
        'updates': None,
        'optimizer': None,
        'wandb_id': None,
        'date': None,
        'week16_note': 'COCO128 baseline-compatible transfer + one2many to one2one copy initialization',
    }
    torch.save(ckpt_out, out)

    report = {
        'source_weights': args.source_weights,
        'out_weights': str(out),
        'out_weights_size': out.stat().st_size,
        'source_state_items': len(src_state),
        'target_state_items': len(dst_state),
        'transferred_items': len(compatible),
        'branch_copy_report': branch_report,
    }
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
