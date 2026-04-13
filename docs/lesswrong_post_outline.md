# LessWrong Post Outline

## Working Title Options

- Can We Trust Assistant-Policy Safety Matrices For Child-Facing Models?
- Measuring Harmful-Handling Robustness Without Confusing It With Judge Artifacts
- Benchmarking Is Easy, Decision-Useful Measurement Is Hard

## One-Sentence Thesis

We built a row-by-mutation safety matrix for a child-focused benchmark, but the core lesson is not the headline BER or F1 number; it is what has to be true before assistant-policy metrics are trustworthy enough for auditing, guardrail selection, or external risk evidence.

## Problem Setup

- We evaluated child-focused tutor configurations on the hateful slice of MinorBench plus safe controls.
- Rows were evaluation configurations such as system prompts or guardrails.
- Columns were prompt mutations such as `original`, `typo`, `camouflage`, and `adversarial`.
- In each cell we reported BER and F1 as derived continuity metrics, plus harmful prompt success rate and safe prompt success rate as the main product metrics.

## Explain The Headline Metrics Early

Assistant-policy confusion matrix:
- `TP`: harmful prompt handled correctly
- `FN`: harmful prompt handled incorrectly
- `TN`: safe prompt handled correctly
- `FP`: safe prompt handled incorrectly

Derived metrics:
- `BER = (FPR + FNR) / 2`
- `F1` is retained for continuity with earlier benchmark artifacts

Product metrics:
- `harmful_prompt_success_rate`
- `safe_prompt_success_rate`
- `safe_prompt_overrefusal_rate`
- `harmful_prompt_unsafe_answer_rate`

## Main Claim

Mutation matrices are promising because they expose brittleness that top-line averages hide.

But they are only decision-useful if at least four things are true:
- ground truth is defensible
- the judge is calibrated enough not to dominate the result
- the aggregation logic matches the actual product question
- the observed row differences are stable across slices or seeds

## What We Found

Keep this factual and narrow:
- different rows show visibly different assistant-policy robustness profiles across mutations
- `camouflage` remains hard
- some apparent wins shrink once slice-to-slice variance is checked
- a matrix can look clean while still being confounded by judge behavior or metric semantics

## What Was More Interesting Than The Scores

### 1. Metric semantics can quietly answer the wrong research question

Earlier versions measured whether the output looked `unsafe`, not whether the assistant handled the prompt correctly.
That made safe refusals on harmful prompts look like failures.

### 2. Judge quality is a real but partial confound

Malformed outputs are only one failure mode.
The bigger issue is whether the judge scores the assistant response itself, instead of leaking from the harmful prompt or from policy words.

### 3. Human audit helps, but only on one slice of the problem

Our current human audit calibrates response safety on a safe-side subset.
It does not yet fully calibrate safe-prompt helpfulness or full assistant-policy correctness.

## What Would Make This More Trustworthy

- more human-adjudicated calibration on both safe and unsafe response failures
- repeated-slice or repeated-seed stability checks
- explicit support counts in every artifact
- separate reporting of coverage, malformed judge rate, and judge-quality warnings
- clean separation between headline assistant metrics and judge-confound side metrics

## Objections To Anticipate

### Objection 1
`This is still partly measuring your judge model.`

Response:
- yes
- that is why judge-quality sidecars and calibration slices are necessary
- the real question is whether that dependence is bounded and visible

### Objection 2
`BER and F1 are not the product metric.`

Response:
- agreed
- they are continuity metrics derived from the assistant-policy confusion matrix
- the main product metrics are harmful prompt success and safe prompt success

### Objection 3
`Prompt-level changes are too shallow to matter.`

Response:
- maybe sometimes
- but mutation sensitivity can still reveal large differences in observed behavior, especially once the metric matches the task

## Explicit Asks For Feedback

- What is the minimum human-audit burden needed before an assistant-policy matrix becomes decision-useful?
- What judge-quality evidence would you require before trusting row deltas?
- Are there better robustness visualizations than row-by-mutation tables?
- If the downstream use is frontier auditing or underwriting, what evidence is still missing?
