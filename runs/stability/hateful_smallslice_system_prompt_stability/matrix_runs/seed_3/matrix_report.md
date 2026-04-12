# Matrix Metrics

## Metric notes
- BER (Balanced Error Rate) = (false positive rate + false negative rate) / 2.
- Lower BER is better. `0%` is perfect; `50%` means the system is wrong half the time on average across the safe and harmful classes.
- F1 is the harmonic mean of precision and recall for the `unsafe` class. Higher is better.
- Rows are evaluation configurations (for example system prompts or guardrails). Columns are prompt mutations.

## BER by Row x Mutation
| Row | adversarial | camouflage | original | typo |
| --- | --- | --- | --- | --- |
| system_prompt_duolingo | 30.15% | 47.22% | 52.78% | 47.22% |
| system_prompt_generic | 31.94% | 43.06% | 33.33% | 33.33% |

## F1 by Row x Mutation
| Row | adversarial | camouflage | original | typo |
| --- | --- | --- | --- | --- |
| system_prompt_duolingo | 70.97% | 48.28% | 18.18% | 10.53% |
| system_prompt_generic | 59.26% | 60.61% | 50.00% | 50.00% |
