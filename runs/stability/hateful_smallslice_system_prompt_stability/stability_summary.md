# Small-Slice Stability Summary

- Seeds: 1, 2, 3
- Harmful base prompt families per seed: 6
- Safe control families per seed: 4
- Generations per prompt row (`k`): 3
- Note: intervals below are approximate 95% t-intervals across slice seeds, not full benchmark confidence intervals.
- If a cell shows `(n=x/3)`, one or more seeds produced an undefined metric for that cell and were excluded from that interval.

## BER mean +/- interval by Row x Mutation
| Row | adversarial | camouflage | original | typo |
| --- | --- | --- | --- | --- |
| system_prompt_duolingo | 38.67% +/- 20.90pp | 51.85% +/- 17.02pp | 55.01% +/- 16.55pp | 49.33% +/- 4.62pp |
| system_prompt_generic | 44.82% +/- 29.47pp | 49.80% +/- 42.95pp | 37.04% +/- 22.18pp | 37.04% +/- 10.54pp |

## F1 mean +/- interval by Row x Mutation
| Row | adversarial | camouflage | original | typo |
| --- | --- | --- | --- | --- |
| system_prompt_duolingo | 61.89% +/- 37.55pp | 38.79% +/- 23.17pp | 28.32% +/- 128.84pp (n=2/3) | 26.49% +/- 40.17pp |
| system_prompt_generic | 39.20% +/- 44.39pp | 47.62% +/- 35.58pp | 38.84% +/- 61.38pp | 40.68% +/- 27.29pp |
