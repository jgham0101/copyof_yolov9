
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--drive', required=True); args=ap.parse_args(); root=Path(args.drive)
    required=['week15_experiment_config.json','selected_reference_weights.json','one2one_copy_init/week15_from_baseline_one2onecopy_init.pt','one2one_copy_init/one2one_copy_init_report.json','sweeps/week15_one2one_copy_init_val_e2e_sweep.csv','sweeps/week15_detach_off_from_baseline_val_e2e_sweep.csv','week15_diagnosis_report/week15_diagnosis_summary.md','week15_diagnosis_report/week15_evidence.json','week15_diagnosis_report/week15_all_sweeps_with_week14_reference.csv','week15_diagnosis_report/week15_best_by_tag.csv','logs/one2one_copy_init_train.log','logs/detach_off_from_baseline_train.log','runs_backup','backup_files']
    for rel in required:
        p=root/rel; assert p.exists(), f'missing: {p}'; print('OK:', p)
    for name in ['week15_one2one_copy_init_val_e2e_sweep.csv','week15_detach_off_from_baseline_val_e2e_sweep.csv']:
        df=pd.read_csv(root/'sweeps'/name)
        assert len(df)>=10, name
        assert {'nms','no-nms'}.issubset(set(df['postprocess'].dropna())), name
        for col in ['precision','recall','map50','map50_95']:
            assert col in df.columns and df[col].notna().all(), f'{name}: {col}'
    best=pd.read_csv(root/'week15_diagnosis_report/week15_best_by_tag.csv')
    assert len(best)>=2
    print('Week 15 output sanity check passed')
if __name__=='__main__': main()
