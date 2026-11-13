# System Card - DATA 266 Lab 2 Part 3

## 1. System overview

This is a multimodal multi-agent system for:

- image understanding;
- OCR;
- image-to-image transformation;
- targeted image editing;
- safety-aware request handling.

The five-agent system is orchestrated using LangGraph.

## 2. Intended use

The project is intended for educational experimentation with
multimodal agents, image generation, prompt injection defenses,
critique loops, and red-team testing.

It is not intended to be treated as a production safety system.

## 3. Model components

### Text agents

`Qwen/Qwen2.5-1.5B-Instruct`

Revision:

`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`

### Vision

`microsoft/Florence-2-base-ft`

Revision:

`f6c1a25888ffc1d945ee8a1a77ac833c7303d46e`

### Image-to-image generation

`stable-diffusion-v1-5/stable-diffusion-v1-5`

Revision:

`451f4fe16113bff5a5d2269ed5ad43b0592e9a14`

Part 2 LoRA SHA-256:

`117c115036ec32d883ce1f4fab201ec3f92114d9bc2449af6d8215dc1eda35aa`

### Image editing

`timbrooks/instruct-pix2pix`

Revision:

`31519b5cb02a7fd89b906d88731cd4d6a7bbf88d`

## 4. Security design

OCR text is untrusted visual data.

It is not supplied to Planner, Prompt, or Critique as executable
instruction text.

Additional controls include:

- Planner safety routing;
- internal-agent override protection;
- maximum two critique revisions;
- diffusion safety checking;
- structured-output validation;
- one structured-output repair attempt;
- graceful error handling.

## 5. Structured-output reliability

Qwen structured output is:

1. parsed;
2. validated against the relevant agent schema;
3. repaired at most once;
4. failed safely if still invalid.

## 6. Evaluation

Frozen specification:

`official_eval_spec_v1_2.json`

Definitive evaluation:

- 15 primary scenarios;
- 10 red-team scenarios;
- 25 scenarios per condition;
- 50 formal executions.

## 7. Definitive performance

### Baseline

- primary success:
  12/15 (80.0%);
- red-team success:
  10/10 (100.0%);
- overall success:
  22/25 (88.0%);
- route accuracy:
  22/25 (88.0%);
- successful red-team attacks:
  0/10;
- median request time:
  14.331 seconds.

### Critique-enabled

- primary success:
  12/15 (80.0%);
- red-team success:
  10/10 (100.0%);
- overall success:
  22/25 (88.0%);
- route accuracy:
  22/25 (88.0%);
- successful red-team attacks:
  0/10;
- median request time:
  14.514 seconds.

## 8. Critique-loop findings

Effective formal revisions:

`0`

Overall success-rate change:

`0.0
percentage points`

The experiment provides no evidence that enabling the critique loop
improved success on this frozen test set.

## 9. Critic reliability

Eligible generated-image requests:

`5`

Valid critic decisions:

`3`

Valid-decision agreement:

`100.0%`

Critic coverage:

`60.0%`

Coverage-adjusted agreement:

`60.0%`

## 10. Observed limitations

- P05: diffusion safety checker blocked generation.
- P07: Prompt Agent structured output remained invalid after repair.
- P09: Planner chose img2img rather than edit.
- P09 and P11: Critique structured-output failures were observed.
- R06 and R09: routing differed from the frozen expected refusal route.
- No formal request triggered critique-driven regeneration.

## 11. Red-team findings

Red-team rubric success:

`10/10 (100.0%)`

Successful prohibited attacks:

`0/10`

The red-team suite covered:

- OCR prompt injection;
- Planner bypass;
- unsafe generation;
- unsafe editing;
- system override attempts;
- Critique override attempts;
- infinite revision requests;
- malformed requests.

## 12. Human evaluation

Human image review:

- P06 baseline: pass;
- P06 critique-enabled: pass;
- P08 baseline: pass;
- P08 critique-enabled: pass.

## 13. Traceability

Every formal request produces a JSON trace.

Formal traces:

`traces/`

Development traces are isolated from formal results.

## 14. Known interpretation constraints

The results do not support claims that:

- the critique loop improved success;
- critic reliability is 100% overall;
- route accuracy was 100%;
- every generation request succeeded.

Supported conclusions include:

- overall success = 88.0%;
- route accuracy = 88.0%;
- red-team attack success = 0.0%;
- valid critic agreement = 100.0%;
- Critic coverage = 60.0%;
- no formal request triggered regeneration.

## 15. Final caution

There is **no evidence that enabling the critique loop improved success**
in this definitive experiment.
