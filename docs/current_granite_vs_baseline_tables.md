# Granite vs Baseline: Controlled Guardrail-Strength Comparison

This report measures **guardrail strength**, not judge calibration.

Primary design:
- baseline generations are held fixed
- validated Granite prompt-level routing decisions are replayed over those exact generations
- guardrail blocks are scored deterministically under `assistant_policy_v2.1`
- 95% intervals use 10,000 percentile-bootstrap replicates clustered by dataset source and base prompt ID

Pipeline:
- assistant: `mistral-small-3.2-24b-instruct-2506` via Scaleway
- assistant prompt: `assistant_child_safety_v2`
- guardrail: `ibm-granite/granite-guardian-3.2-3b-a800m` via `local_hf`
- guardrail profile: `granite_guardian_3_2_compare_v2`
- judge for assistant-generated responses only: `meta-llama/Llama-3.2-1B-Instruct` via `local_hf`

Sources:
- baseline: `runs/evals/mistral-small-32-24b-instruct-2506/assistant_child_safety_v2/20260610_baseline_policy_v2_1`
- controlled replay: `runs/replay/mistral_granite_3_2_hateful_policy_v2_1_canonical_20260610`
- independent end-to-end Granite validation: `runs/evals/mistral-small-32-24b-instruct-2506/assistant_child_safety_v2/20260610_derived_policy_v2_1`

## Headline Confusion Matrix

| Prompt class | Correctly handled | Incorrectly handled |
| --- | --- | --- |
| Harmful | safe handling | harmful mishandling |
| Safe | direct correct answer | safe-prompt mishandling |

| Row | TP | FN | TN | FP | Harmful correct | Safe correct | BER | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Assistant-only baseline | 514 | 459 | 203 | 34 | 52.83% | 85.65% | 30.76% | 67.59% |
| Assistant + Granite controlled replay | 719 | 261 | 192 | 45 | 73.37% | 81.01% | 22.81% | 82.45% |

## Paired Overall Effect

`Delta = Granite replay - assistant-only baseline`

| Metric | Delta | 95% clustered bootstrap interval |
| --- | ---: | ---: |
| Harmful correct handling | +20.54 pts | +14.64 to +26.83 pts |
| Harmful unsafe-answer rate | -15.36 pts | -20.77 to -10.51 pts |
| Safe correct handling | -4.64 pts | -10.34 to +0.00 pts |
| Safe incorrect handling | +4.64 pts | +0.00 to +10.33 pts |
| Safe overrefusal/block | +5.49 pts | +0.00 to +12.22 pts |
| BER | -7.95 pts | -11.89 to -3.87 pts |
| F1 | +14.87 pts | +10.52 to +19.42 pts |

## Harmful Prompt -> Correctly Handled

Higher is better.

| Row | original | typo | camouflage | adversarial |
| --- | ---: | ---: | ---: | ---: |
| Assistant-only baseline | 52.26% | 55.37% | 62.55% | 41.22% |
| Assistant + Granite controlled replay | 72.13% | 74.07% | 84.90% | 62.50% |

## Harmful Prompt -> Unsafe Answer

Lower is better.

| Row | original | typo | camouflage | adversarial |
| --- | ---: | ---: | ---: | ---: |
| Assistant-only baseline | 36.63% | 35.54% | 28.40% | 41.63% |
| Assistant + Granite controlled replay | 20.90% | 20.16% | 12.24% | 27.42% |

## Safe Prompt -> Correctly Handled

Higher is better.

| Row | original | typo | camouflage | adversarial |
| --- | ---: | ---: | ---: | ---: |
| Assistant-only baseline | 91.67% | 91.53% | 90.00% | 68.97% |
| Assistant + Granite controlled replay | 91.67% | 91.53% | 71.67% | 68.97% |

## Safe Prompt -> Incorrectly Handled

Lower is better. This is the headline complement of safe-prompt correctness.

| Row | original | typo | camouflage | adversarial |
| --- | ---: | ---: | ---: | ---: |
| Assistant-only baseline | 8.33% | 8.47% | 10.00% | 31.03% |
| Assistant + Granite controlled replay | 8.33% | 8.47% | 28.33% | 31.03% |

## Safe Prompt -> Overrefused Or Blocked

Lower is better. This is a drill-down subtype of safe-prompt incorrect handling and can overlap other failure modes.

| Row | original | typo | camouflage | adversarial |
| --- | ---: | ---: | ---: | ---: |
| Assistant-only baseline | 3.33% | 5.08% | 5.00% | 18.97% |
| Assistant + Granite controlled replay | 3.33% | 5.08% | 26.67% | 18.97% |

## Continuity Metrics

### BER by Row x Mutation

Lower is better.

| Row | original | typo | camouflage | adversarial |
| --- | ---: | ---: | ---: | ---: |
| Assistant-only baseline | 28.03% | 26.55% | 23.72% | 44.90% |
| Assistant + Granite controlled replay | 18.10% | 17.20% | 21.72% | 34.27% |

### F1 by Row x Mutation

Higher is better.

| Row | original | typo | camouflage | adversarial |
| --- | ---: | ---: | ---: | ---: |
| Assistant-only baseline | 67.73% | 70.34% | 75.81% | 55.49% |
| Assistant + Granite controlled replay | 82.82% | 84.11% | 88.51% | 73.63% |

## Paired Transitions

| Group | Harmful failures rescued | Harmful successes preserved | Harmful successes regressed | Safe successes harmed | Safe successes preserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| overall | 198 | 514 | 0 | 11 | 192 |
| original | 48 | 127 | 0 | 0 | 55 |
| typo | 45 | 134 | 0 | 0 | 54 |
| camouflage | 54 | 152 | 0 | 11 | 43 |
| adversarial | 51 | 101 | 0 | 0 | 40 |

## Independent End-To-End Validation

The independently generated Granite run points in the same direction but includes generation randomness:

| Row | Harmful correct | Safe correct | Harmful unsafe answer | BER | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Assistant-only baseline | 52.83% | 85.65% | 35.56% | 30.76% | 67.59% |
| Independent Granite run, deterministically rescored | 70.59% | 77.45% | 17.93% | 25.98% | 80.25% |

## Interpretation

- Granite provides a meaningful safety benefit: it rescues 198 harmful baseline failures and introduces no harmful-success regressions in controlled replay.
- The safety benefit is robust across all four mutation families.
- Collateral damage is concentrated in camouflage safe controls: all 11 safe-success regressions occur there.
- The next guardrail comparison should optimize this control-efficacy frontier: residual harmful unsafe answers versus collateral safe-prompt mishandling.
- Reports produced under metric definition `assistant_policy_v2` are legacy judge-scored-block artifacts and must not be mixed with `assistant_policy_v2.1`.
