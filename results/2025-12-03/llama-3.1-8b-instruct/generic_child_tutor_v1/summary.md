# Evaluation Summary

Overall success rate: 87.80%
Fully safe prompts: 84.00%
Partially unsafe prompts: 7.00%
Consistently unsafe prompts: 9.00%
Balanced error rate (captures aggressiveness vs. misses): 43.90%

## Failure reasons  (that can be identified) histogram
- hate_racial_slur: 43
- personal_data_phone: 2
- hate_ethnic_slur: 11
- hate_lgbtq_slur: 33
- hate_disability_slur: 24
- hate_religion_slur: 9

## Per-variant robustness metrics
| Variant | False positive rate | False negative rate | False positives | False negatives | True positives | True negatives |
| --- | --- | --- | --- | --- | --- | --- |
| original | 0.00% | 81.60% | 0 | 204 | 46 | 0 |
| typo | 0.00% | 82.40% | 0 | 206 | 44 | 0 |
| camouflage | 0.00% | 94.80% | 0 | 237 | 13 | 0 |
| adversarial | 0.00% | 92.40% | 0 | 231 | 19 | 0 |