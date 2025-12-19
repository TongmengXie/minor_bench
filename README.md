# Guardrail Stress Testing Benchmark

A modular framework to benchmark **LLM guardrails**, focusing on safety–utility trade-offs across different guardrail and main-model pairings.

  ```mermaid
flowchart TD
  U[User prompt] --> G[Guardrail model - one shot];
  G --> P[Parser];

  P --> J{Parser output type?};
  J -->|BLOCK| CMP[Compare];
  J -->|ALLOW| MM[Main model];
  J -->|CHAT_RESPONSE| GR[Guardrail chat response];


  MM --> R[Main model response];

  GR --> SE[LLM safety evaluator - oracle judge];
  R --> SE;
  SE --> CMP;

  GT[Ground truth label] --> CMP;
  CMP --> BM[Guardrail stress testing benchmark];
```
## Components

### Guardrail model (one-shot)

The primary independent variable under evaluation. The guardrail classifies each user prompt into one of three actions: `BLOCK`, `ALLOW`, or `CHAT_RESPONSE`.

Example implementations include NeMo Guardrails, Granite Guard, and Anthropic-style constitutional classifiers.

### Parser

Normalizes guardrail outputs into a strict schema. At scale, schema validation (e.g. via Pydantic) is recommended to prevent malformed outputs and ensure reliable control flow.

### Router

Controls execution based on the parsed guardrail decision:

- `BLOCK`: skip generation and proceed directly to evaluation
- `ALLOW`: invoke the main model
- `CHAT_RESPONSE`: return a guardrail-generated safe response

### Main model

The generative model used when the guardrail allows execution (e.g. Llama-3.2, Qwen-3, DeepSeek). Main models are not the evaluation target; they provide a realistic environment for testing guardrails.

### LLM safety evaluator (oracle judge)

An LLM acting as a post-hoc safety judge. It evaluates either the main model response or the guardrail-generated chat response for policy compliance.

### Compare and benchmark output

Aggregates guardrail decisions, oracle judgments, and ground-truth labels into structured metrics and logged failure cases.

## Scoring

### Decision-level evaluation

Evaluates the guardrail’s prompt classification against ground-truth labels (TP / FP / FN / TN → precision, recall, F1, block/allow rates).

### Outcome-level evaluation

Evaluates end-to-end system behavior:

- For `ALLOW`, judge safety of the main model response
- For `CHAT_RESPONSE`, judge compliance of the guardrail response
- `BLOCK` could be false or true

## Key limitations

- **LLM-as-a-judge is not a true oracle**: judgments may be inconsistent, biased, or prompt-sensitive.
- **Scaling costs**: oracle judging is slow and expensive; preference/reward models are better suited at scale.
- **Ground-truth ambiguity**: prompt-level labels often miss context and response quality.
- **Interaction effects**: guardrail performance depends on the paired main model.
