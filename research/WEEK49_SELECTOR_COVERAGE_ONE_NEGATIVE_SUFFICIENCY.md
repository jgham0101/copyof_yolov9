# Week49 — Selector Coverage vs One-Negative Sufficiency

## Validity
- Training performed: **False**
- A/B reproduction gate: **True**
- Raw O2O ↔ canonical proxy anchor-score alignment: **PASS**

## Control selector coverage
- duplicate pairs/image: 256.3424
- selected duplicate-key coverage: 4.31%
- eligible duplicate-key coverage: 16.63%
- eligible endpoint-pair coverage: 29.29%

## Treatment selector coverage
- duplicate pairs/image: 253.0636
- selected duplicate-key coverage: 4.11%
- eligible duplicate-key coverage: 16.29%
- eligible endpoint-pair coverage: 28.98%

## Control AP rescue
- SEL1: 31.95%
- SELALL: 40.62%
- COMP1: 63.85%
- COMPALL: 83.41%
- C70: 96.66%

## Treatment AP rescue
- SEL1: 24.32%
- SELALL: 32.50%
- COMP1: 60.34%
- COMPALL: 79.15%
- C70: 96.37%

## Routing
- Control: **MIXED_SELECTOR_COVERAGE_AND_ONE_NEGATIVE_SUFFICIENCY**
- Treatment: **MIXED_SELECTOR_COVERAGE_AND_ONE_NEGATIVE_SUFFICIENCY**
- Consensus: **MIXED_SELECTOR_COVERAGE_AND_ONE_NEGATIVE_SUFFICIENCY**
- Next: **WEEK50_REFINED_UNIQUENESS_DIAGNOSTIC_BEFORE_TRAINING**

All suppression variants are diagnostic only. Problem 1 remains ANALYSIS_COMPLETED_WITH_ACCEPTED_RESIDUAL; strict gate is not relaxed.

## Method contract
- No training in Week49.
- Exact Week48 selector replay from raw O2O assigner/BCE state.
- Indexed no-NMS retains anchor/class ids from canonical proxy.
- SEL1/SELALL/COMP1/COMPALL/C70 are diagnostic only.
- Interpretation blocked if A/B reproduction fails.
