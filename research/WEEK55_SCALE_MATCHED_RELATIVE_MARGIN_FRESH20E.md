# Week55 — Scale-Matched Relative-Margin Fresh20e

- Frozen lambda before AP: **0.279462**
- Scale validation effective/absolute ratio: **1.0969**
- Control retrained: **False**
- Treatment fresh20e: **True**

## Frozen control
- +NMS AP: 0.428112
- no-NMS AP: 0.410260
- dup/image: 256.4206

## Week52 absolute multi-rival
- +NMS AP: 0.425122
- no-NMS AP: 0.407692
- dup/image: 254.1248

## Week55 scale-matched relative margin
- +NMS AP: 0.426333
- no-NMS AP: 0.408410
- dup/image: 238.0762
- C70 AP: 0.425852

## Week55 - control
- +NMS AP delta: -0.001779
- no-NMS AP delta: -0.001851
- duplicate reduction: 7.1540%
- margin delta: +0.001340
- selected rival relative change: -2.33%

- Verdict: **SCALE_MATCHED_RELATIVE_MARGIN_NOT_CAUSALLY_SUFFICIENT**
- Next: **STOP_OR_FINAL_LIMITATION_ANALYSIS**

## Causal contract
- Pre-AP frozen lambda: `0.279461747770`
- Pretrain freeze HEAD: `4a76d5fb95b59aa68de99430ca123b1e5b1b521d`
- Same Week33 init / data / seed / epochs as control.
- Week51 selector unchanged.
- Relative margin fixed at 1.0 logit.
- Control was not retrained.
