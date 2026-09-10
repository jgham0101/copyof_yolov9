# Week47 — Duplicate Uniqueness & Ranking Diagnostic

## Validity
- Training performed: **False**
- Week46 A/B reproduction gate: **PASS**

## Primary A/B
- A native NMS COCO AP50-95: 0.428112
- B no-NMS K300 COCO AP50-95: 0.410260
- Problem2 gap: 0.017852

## Diagnostic suppression rescue
- C50 AP: 0.421646
- C70 AP: 0.427539
- C90 AP: 0.426324
- C70 rescue fraction: 0.9679
- Conclusion: **UNIQUENESS_DUPLICATE_DOMINANT**

## Per-GT multiplicity/ranking
- multiplicity=0 fraction: 0.1419
- multiplicity=1 fraction: 0.1259
- multiplicity=2 fraction: 0.1635
- multiplicity>=3 fraction: 0.5687
- first valid rank median: 6.0
- first valid rank p95: 122.0

## Top-k sensitivity
- K100 AP: 0.408162
- K300 AP: 0.410260
- K500 AP: 0.410523
- K500-K300: +0.000263
- Conclusion: **TOPK_SIZE_NOT_PRIMARY_AP_DRIVER**

## Next
- **WEEK48_UNIQUENESS_CONSISTENCY_TRAINING_INTERVENTION**
- C70 recovers >=80% of the A-B AP gap. Duplicate/uniqueness is the dominant Problem2 failure.
- Fresh training next week: **True**

Diagnostic suppression is not a final deployment method.

## Provenance
- Week47 branch: `week47-duplicate-uniqueness-ranking-diagnostic`
- Start HEAD: `00ad5deab7667fb1228a6c341f443a3f5afaa977`
- Week46 O2O proxy SHA256: `7c9128dccce82394ddeb46b044ff07217841d3f251ab081feba8185505945424`
- No training/fine-tuning performed.
- C50/C70/C90 are diagnostic suppression only.
