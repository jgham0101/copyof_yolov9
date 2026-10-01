# Week54 — Relative-Margin Uniqueness Objective Design

- Full training performed: **False**
- Week53 verdict: **ABSOLUTE_NEGATIVE_BCE_DID_NOT_CREATE_RELATIVE_MARGIN**
- Selector changed: **False**
- Loss: `softplus(rival_logit - stopgrad(rep_logit) + 1.0)`
- Lambda: `1.0`

## Gates
- Minimal one-factor source patch: **True**
- Lambda=0 base equivalence: **True**
- Week51/Week54 selector identity: **True**
- Margin pair activity: **True**
- Margin=1 violation fraction: **0.8957**
- AUX-only gradient isolation: **True**
- 8-batch structural smoke: **True**

- Verdict: **RELATIVE_MARGIN_OBJECTIVE_READY_FOR_FRESH_CAUSAL_TRAINING**
- Next: **WEEK55_ONE_FACTOR_RELATIVE_MARGIN_FRESH20E**
## Causal contract
- No full training in Week54.
- Week51 selector is unchanged.
- Only loss formulation changes to detached-representative relative margin.
- Auxiliary gradient flows only to selected O2O rival class logits.
- Margin is fixed at 1.0 before AP evaluation.
