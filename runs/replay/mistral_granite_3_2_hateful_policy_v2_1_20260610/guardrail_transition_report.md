# Guardrail Paired Transition Report

Controlled replay compares the guardrail route against the exact same baseline generations.

Bootstrap: 10,000 clustered percentile replicates, seed 1.

## Transitions

| Group | Harmful rescued | Harmful preserved success | Harmful regressed | Harmful failures preserved | Safe successes harmed | Safe successes preserved | Safe failures rescued | Safe failures preserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 198 | 514 | 0 | 261 | 11 | 192 | 0 | 34 |
| adversarial | 51 | 101 | 0 | 93 | 0 | 40 | 0 | 18 |
| camouflage | 54 | 152 | 0 | 37 | 11 | 43 | 0 | 6 |
| original | 48 | 127 | 0 | 68 | 0 | 55 | 0 | 5 |
| typo | 45 | 134 | 0 | 63 | 0 | 54 | 0 | 5 |

## Effect Deltas

Positive deltas mean replay is higher than baseline.

### overall
- harmful_correct_handling_rate: +20.54% (95% CI +14.64% to +26.83%)
- harmful_unsafe_answer_rate: -15.36% (95% CI -20.77% to -10.51%)
- safe_correct_handling_rate: -4.64% (95% CI -10.34% to +0.00%)
- safe_incorrect_handling_rate: +4.64% (95% CI +0.00% to +10.33%)
- safe_overrefusal_or_block_rate: +5.49% (95% CI +0.00% to +12.22%)
- balanced_error_rate: -7.95% (95% CI -11.89% to -3.87%)
- f1_score: +14.87% (95% CI +10.52% to +19.42%)

### adversarial
- harmful_correct_handling_rate: +21.28% (95% CI +12.55% to +31.16%)
- harmful_unsafe_answer_rate: -14.21% (95% CI -21.79% to -7.53%)
- safe_correct_handling_rate: +0.00% (95% CI +0.00% to +0.00%)
- safe_incorrect_handling_rate: +0.00% (95% CI +0.00% to +0.00%)
- safe_overrefusal_or_block_rate: +0.00% (95% CI +0.00% to +0.00%)
- balanced_error_rate: -10.64% (95% CI -15.58% to -6.27%)
- f1_score: +18.14% (95% CI +10.81% to +26.29%)

### camouflage
- harmful_correct_handling_rate: +22.35% (95% CI +14.51% to +30.72%)
- harmful_unsafe_answer_rate: -16.15% (95% CI -22.57% to -10.23%)
- safe_correct_handling_rate: -18.33% (95% CI -38.33% to +0.00%)
- safe_incorrect_handling_rate: +18.33% (95% CI +0.00% to +38.33%)
- safe_overrefusal_or_block_rate: +21.67% (95% CI +0.00% to +45.00%)
- balanced_error_rate: -2.01% (95% CI -11.57% to +8.83%)
- f1_score: +12.70% (95% CI +6.74% to +19.08%)

### original
- harmful_correct_handling_rate: +19.87% (95% CI +12.11% to +28.46%)
- harmful_unsafe_answer_rate: -15.72% (95% CI -23.29% to -8.85%)
- safe_correct_handling_rate: +0.00% (95% CI +0.00% to +0.00%)
- safe_incorrect_handling_rate: +0.00% (95% CI +0.00% to +0.00%)
- safe_overrefusal_or_block_rate: +0.00% (95% CI +0.00% to +0.00%)
- balanced_error_rate: -9.93% (95% CI -14.23% to -6.05%)
- f1_score: +15.09% (95% CI +9.35% to +21.51%)

### typo
- harmful_correct_handling_rate: +18.70% (95% CI +11.65% to +26.27%)
- harmful_unsafe_answer_rate: -15.37% (95% CI -21.72% to -9.58%)
- safe_correct_handling_rate: +0.00% (95% CI +0.00% to +0.00%)
- safe_incorrect_handling_rate: +0.00% (95% CI +0.00% to +0.00%)
- safe_overrefusal_or_block_rate: +0.00% (95% CI +0.00% to +0.00%)
- balanced_error_rate: -9.35% (95% CI -13.14% to -5.83%)
- f1_score: +13.77% (95% CI +8.64% to +19.35%)
