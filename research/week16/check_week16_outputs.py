
import argparse, json
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drive', required=True)
    args = ap.parse_args()
    root = Path(args.drive)
    required = [
        'week16_experiment_config.json',
        'data/week16_coco128.yaml',
        'dataset_summary.json',
        'selected_week16_weights.json',
        'one2one_copy_init/week16_from_baseline_one2onecopy_init.pt',
        'one2one_copy_init/one2one_copy_init_report.json',
        'sweeps/proposed_scratch_val_e2e_sweep.csv',
        'sweeps/proposed_from_baseline_val_e2e_sweep.csv',
        'sweeps/proposed_one2one_copy_val_e2e_sweep.csv',
        'week16_diagnosis_report/week16_diagnosis_summary.md',
        'week16_diagnosis_report/week16_best_by_tag.csv',
        'week16_diagnosis_report/week16_all_e2e_sweeps.csv',
        'week16_diagnosis_report/week16_native_val_summary.csv',
        'runs_backup',
        'backup_files',
    ]
    for rel in required:
        p = root / rel
        assert p.exists(), f'missing: {p}'
        print('OK:', p)

    for name in ['proposed_scratch_val_e2e_sweep.csv', 'proposed_from_baseline_val_e2e_sweep.csv', 'proposed_one2one_copy_val_e2e_sweep.csv']:
        df = pd.read_csv(root / 'sweeps' / name)
        assert len(df) >= 10, f'too few rows: {name}'
        assert {'nms', 'no-nms'}.issubset(set(df['postprocess'].dropna())), f'missing postprocess in {name}'
        for col in ['precision', 'recall', 'map50', 'map50_95']:
            assert col in df.columns, f'missing {col} in {name}'
            assert df[col].notna().all(), f'NaN in {col} of {name}'

    best = pd.read_csv(root / 'week16_diagnosis_report/week16_best_by_tag.csv')
    assert {'proposed_scratch', 'proposed_from_baseline', 'proposed_one2one_copy'}.issubset(set(best['tag'].dropna())), 'missing best tags'
    native = pd.read_csv(root / 'week16_diagnosis_report/week16_native_val_summary.csv')
    assert len(native) >= 1, 'native summary is empty'
    print('Week 16 output sanity check passed')

if __name__ == '__main__':
    main()
