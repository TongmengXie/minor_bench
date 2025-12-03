# Evaluation Summary

Overall success rate: 90.00%
Fully safe prompts: 86.50%
Partially unsafe prompts: 6.00%
Consistently unsafe prompts: 7.50%
Balanced error rate (captures aggressiveness vs. misses): 45.00%

## Failure reasons (that can be identified) histogram
- hate_racial_slur: 36
- hate_lgbtq_slur: 30
- hate_disability_slur: 24
- hate_ethnic_slur: 10

## Per-variant robustness metrics
| Variant | False positive rate | False negative rate | False positives | False negatives | True positives | True negatives |
| --- | --- | --- | --- | --- | --- | --- |
| original | 0.00% | 85.20% | 0 | 213 | 37 | 0 |
| typo | 0.00% | 86.40% | 0 | 216 | 34 | 0 |
| camouflage | 0.00% | 99.60% | 0 | 249 | 1 | 0 |
| adversarial | 0.00% | 88.80% | 0 | 222 | 28 | 0 |