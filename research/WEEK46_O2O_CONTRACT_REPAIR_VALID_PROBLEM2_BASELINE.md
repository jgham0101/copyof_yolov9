# Week46 — O2O Contract Repair + Valid Problem 2 Baseline

- Problem1: **ANALYSIS_COMPLETED_WITH_ACCEPTED_RESIDUAL**
- Strict gate: **FAIL (not relaxed)**
- Training: **False**
- Week45 first Problem2 attempt: **not reused**

## Gates
- Gate A semantic proof: **PASS**
- Gate B Week39 reproduction: **PASS**
- Gate C paired native reproduction: **PASS**

## Valid Problem2 comparison
- Internal mAP50-95: NMS 0.428862 -> no-NMS 0.407809 (-0.021053)
- COCO AP50-95: NMS 0.428112 -> no-NMS 0.410260 (-0.017852)
- Precision delta: -0.014782
- Recall delta: -0.034175
- Same-class duplicate pairs/image: 0.033600 -> 256.420600

## Latency
- NMS postprocess: 1.1548 ms
- no-NMS postprocess: 0.7138 ms
- saved: 0.4410 ms

## Three-stage COCO AP50-95
- Week36 baseline +NMS: 0.454000
- accepted-residual O2O +NMS: 0.428112
- same O2O + no-NMS: 0.410260
- Problem1 residual: -0.025888
- Problem2 delta: -0.017852

## Next
- **WEEK47_DUPLICATE_UNIQUENESS_AND_RANKING_DIAGNOSTIC**
- no-NMS increases same-class duplicates and precision falls; diagnose uniqueness/ranking first.
- Retraining: **not before Week47 diagnosis**

## Provenance
- start HEAD: `078550d151dfc4359b4d60e73b605331590bad8a`
- Week39 source SHA256: `f766c69e067981fe24beed0851637710a28e6797b5dd4d5818753faf3eb41586`
- O2O proxy SHA256: `7c9128dccce82394ddeb46b044ff07217841d3f251ab081feba8185505945424`
- no training/fine-tuning
- Week45 invalid Problem2 result not reused
