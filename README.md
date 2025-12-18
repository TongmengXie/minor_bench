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
