# Week51 — Prediction-Space Multi-Rival Uniqueness Objective Design

## Status
- Full training performed: **False**
- Week50 automatic disagreement preserved: **True**
- Progression reason: both control/treatment show the same dominant topology and monotonic R1<R2<R3 rescue; disagreement was a heuristic boundary effect.

## Objective
- Prediction-space top-K anchors: **300**
- Same predicted class: **True**
- Direct predicted-box IoU threshold: **0.70**
- Greedy score ordering: **True**
- Max rivals/group: **3**
- O2O foreground rivals protected: **True**
- Selection geometry/ranking: **stop-gradient**
- Rival penalty: **BCE target=0 / softplus**
- Normalization: **mean within group, mean across groups**
- Auxiliary gradient: **O2O classification only**

## Gates
- Lambda=0 base equivalence: **True**
- Auxiliary gradient isolation: **True**
- 8-batch structural activity: **True**
- R1/R3 normalization sanity: **True**
- Mean rivals/group (first gradient batch): **2.3516**
- 8-batch weighted mean rivals/group: **2.3196**
- Zero-sameclass-positive group fraction (8 batches): **0.8404**
- Unmatched representative group fraction (8 batches): **0.9174**

## Verdict
**PREDSPACE_MULTI_RIVAL_OBJECTIVE_READY_FOR_FRESH_CAUSAL_TRAINING**

## Next
**WEEK52_ONE_FACTOR_PREDSPACE_MULTI_RIVAL_FRESH20E**

Week51 does not claim AP improvement. It only establishes a causally motivated and gradient-safe treatment for the next fresh training ablation.
## Causal contract
- Week51 performs no full training.
- Future treatment base is Week39 cls×2, not Week48 treatment stacking.
- Only new factor is prediction-space greedy multi-rival O2O classification-negative auxiliary.
- Box geometry and all selection decisions are stop-gradient.
- O2O matched foreground anchors are never auxiliary rival negatives.
- Auxiliary gradient must remain isolated to one2one_cv3.
