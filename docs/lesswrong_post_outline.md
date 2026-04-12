# LessWrong Post Outline

## Working Title Options

- Can We Trust Safety Robustness Matrices For Child-Facing Models?
- Measuring Safety Robustness By Mutation: What BER Matrices Tell Us And What They Do Not
- Benchmarking Is Easy, Trustworthy Measurement Is Hard

## One-Sentence Thesis

We built a row-by-mutation safety matrix for a child-focused benchmark, but the main lesson is not the headline score; it is what has to be true before such a matrix is trustworthy enough for auditing, guardrail evaluation, or risk underwriting.

## Target Reader

- alignment researchers
- evals people
- frontier AI auditors
- people interested in model measurement for external governance

## Opening Hook

Most benchmark posts implicitly ask:
`Which model or guardrail is better?`

The more important question is:
`When should we trust the measurement enough to make decisions from it?`

## Problem Setup

- We evaluated a child-focused tutor model on the hateful slice of MinorBench.
- Rows were evaluation configurations such as system prompts.
- Columns were prompt mutations such as `original`, `typo`, `camouflage`, and `adversarial`.
- In each cell we reported BER and F1.

## Explain BER Early

`BER = (false positive rate + false negative rate) / 2`

Why use it:
- It weights errors on safe prompts and harmful prompts equally.
- That matters because the benchmark is class-imbalanced.
- Lower BER is better.

## The Artifact

Include:
- one matrix screenshot or table
- one example row summary
- one short paragraph defining the labels and aggregation unit

Reference artifact:
- [matrix_report.md](/root/minor_bench/runs/matrix/hateful_system_prompt_rows_stable_20260302_132626/matrix_report.md)
- [small-slice stability summary](/root/minor_bench/runs/stability/hateful_smallslice_system_prompt_stability/stability_summary.md)

## Main Claim

Mutation matrices are promising because they expose brittleness that top-line safety rates hide.

But they are only decision-useful if at least four things are true:
- ground truth is defensible
- the judge is calibrated
- the aggregation logic is faithful to the product question
- the observed row differences are stable across slices

## What We Found

Keep this factual and narrow:
- Different system prompts show visibly different BER/F1 profiles across mutations.
- `camouflage` tends to be hard.
- Some apparent wins disappear or become uncertain once you check slice-to-slice variance.
- Small-slice 3-seed intervals are wide, so not every cell difference is robust.

## What Was More Interesting Than The Scores

### 1. Reporting Bugs Can Create Fake Interpretation

Example:
- no-guardrail runs were initially displayed as `Guardrail: unknown (scaleway)`
- that was a reporting bug, not a substantive finding

Lesson:
- measurement pipelines need audit trails, not just final scores

### 2. Judge Quality Is A Real But Partial Confound

Important nuance:
- malformed judge outputs matter
- but low malformed rate alone does not guarantee validity
- the bigger issue is whether the judge rubric matches the intended child-safety standard

### 3. Stability Matters More Than A Single Pretty Matrix

The small-slice 3-seed run is the right kind of embarrassment:
- it tells us that some differences are not yet stable enough to overinterpret

## What Would Make This More Trustworthy

List concrete requirements:
- human-adjudicated calibration slice
- disagreement analysis on false positives and false negatives
- repeated-slice or repeated-seed stability
- explicit support counts in every artifact
- separate reporting of measurement coverage and malformed-judge rate

## Why This Might Matter Beyond Benchmarking

This is where the ExonaLab / audit angle fits.

If we want benchmark outputs to be useful for:
- guardrail selection
- external auditing
- insurer-facing risk evidence

then the deliverable cannot just be a score.
It has to be a score plus evidence that the score is meaningful.

## Strongest Objections To Anticipate

### Objection 1
`This is just measuring your judge model.`

Response:
- partly true
- that is why we track malformed coverage and now propose human calibration
- the right question is not whether the judge matters, but whether we can bound and measure that dependence

### Objection 2
`BER and F1 are not the product metric.`

Response:
- agreed
- they are intermediate measurement tools
- the real question is which risk-sensitive decision they are supposed to support

### Objection 3
`System prompts are too shallow to matter.`

Response:
- maybe
- but the observed mutation sensitivity suggests prompt-level policy still changes output behavior enough to warrant measurement

## Explicit Asks For Feedback

Ask the community:
- What is the minimum human-audit burden needed before such a matrix becomes decision-useful?
- What would you want to see before trusting BER/F1 deltas between rows?
- Are there better ways to represent robustness than row-by-mutation matrices?
- If the downstream use is frontier auditing or underwriting, what evidence is still missing?

## Optional Closing

My current view is that robustness matrices are useful, but only as part of a measurement stack.
The interesting work is not just producing them.
It is making them hard to fool, hard to misread, and cheap enough to rerun.

