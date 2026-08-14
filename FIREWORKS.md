# Fireworks AI in the PCB Self-Improving Agent

Fireworks is the primary intelligence layer for this project. Every decision the
agent makes goes through it: what is wrong with the classifier, which experiment
to run next, whether that experiment worked, and what lesson to leave behind for
the next run. It also supplies the vectors that MongoDB Atlas searches, the
reranker that keeps retrieval honest, and the vision models that look at the PCB
crops the classifier got wrong.

Everything described here lives in [`pcb_agent/fireworks/`](./pcb_agent/fireworks/)
and is exercised by 157 offline tests plus a live smoke test.

---

## Contents

- [Quick start](#quick-start)
- [What is integrated](#what-is-integrated)
- [Why each capability is used](#why-each-capability-is-used)
- [Setup](#setup)
- [The model registry](#the-model-registry)
- [Capability guide](#capability-guide)
  - [Chat completions](#1-chat-completions)
  - [Structured outputs](#2-structured-outputs-json-schema)
  - [Tool calling](#3-tool-calling-the-experiment-allowlist)
  - [Embeddings](#4-embeddings-for-atlas-vector-search)
  - [Reranking](#5-reranking)
  - [Vision](#6-vision-inspecting-misclassified-crops)
  - [Streaming](#7-streaming)
  - [Grammar mode](#8-grammar-mode)
  - [Prompt caching](#9-prompt-caching)
  - [Reasoning effort](#10-reasoning-effort)
- [Cost and token accounting](#cost-and-token-accounting)
- [Reliability](#reliability)
- [LangChain and LangGraph](#langchain-and-langgraph)
- [Mapping to the 3-person plan](#mapping-to-the-3-person-plan)
- [Module reference](#module-reference)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Not yet integrated](#not-yet-integrated)

---

## Quick start

```bash
cp .env.example .env          # fill in FIREWORKS_API_KEY
pip install -r requirements.txt

# 1. Does everything work with your key today?
python scripts/fireworks_smoke.py

# 2. Run the full autonomous loop (cold start, then experienced)
python -m pcb_agent.demo.fireworks_loop --run a --budget 5 --fresh
python -m pcb_agent.demo.fireworks_loop --run b --budget 5
```

In code:

```python
from pcb_agent.fireworks import FireworksClient, diagnose, propose_experiment, critique_result

with FireworksClient() as client:
    d = diagnose(client, state, memories)                 # schema-constrained JSON
    p = propose_experiment(client, state, d, memories=memories)   # forced tool call
    # ... execute p, re-evaluate ...
    c = critique_result(client, before, after, p)         # verdict + reusable lesson
    print(client.ledger.summary_line())
    # fireworks: 3 calls, 8104 tokens, 4.2s, 0 retries
```

---

## What is integrated

| Fireworks capability | Endpoint / parameter | Used for | Module |
|---|---|---|---|
| Chat completions | `POST /chat/completions` | every reasoning step | `client.py` |
| Structured outputs | `response_format: json_schema` | diagnosis, critique, lessons, vision reports | `structured.py` |
| Tool / function calling | `tools` + `tool_choice: required` | choosing an executable experiment | `tools.py` |
| Parallel tool calls (disabled) | `parallel_tool_calls: false` | forcing one change at a time | `scientist.py` |
| Forced tool selection | `tool_choice: {function: ...}` | pinning an action for A/B tests | `tools.force_tool` |
| Grammar mode (BNF) | `response_format: grammar` | single-token verdicts | `structured.VERDICT_GRAMMAR` |
| Embeddings | `POST /embeddings` | Atlas Vector Search vectors | `embeddings.py` |
| Matryoshka truncation | `dimensions` | 1024-dim vectors, smaller index | `embeddings.py` |
| Reranking | `POST /rerank` | precision pass over vector recall | `rerank.py` |
| Vision / multimodal | `image_url` content parts | inspecting misclassified crops | `vision.py` |
| Streaming | `stream` + `stream_options` | live demo console | `client.stream_text` |
| Prompt caching | `prompt_cache_isolation_key` | per-run cache isolation | `config.py` |
| Reasoning effort | `reasoning_effort` | escalation on hard diagnoses | `config.py` |
| Perf metrics | `perf_metrics_in_response` | server-side latency breakdown | `telemetry.py` |
| Seeded sampling | `seed` | reproducible runs | `client.chat` |
| Model catalogue | `GET /models` | detecting retired model ids | `models.py` |
| Usage accounting | `usage` + cached token details | credit budget, per-run cost | `telemetry.py` |
| OpenAI compatibility | base URL swap | LangChain / LangGraph nodes | `langchain_adapters.py` |

Not every one is load-bearing. The essential four are structured outputs, tool
calling, embeddings, and reranking. The rest are there because the project has
$50 of Fireworks credit and specific problems those features solve.

---

## Why each capability is used

The brief asks for something narrower than "call an LLM a lot": the agent must
*choose executable experiments* and *get better across runs*. Three of the
integrations exist specifically to make that claim survive scrutiny.

**Tool calling, not prose.** The scientist calls
`propose_experiment` with `tool_choice="required"` and
`parallel_tool_calls=False`. The model cannot answer in text, and it cannot
propose two changes at once. Every experiment in the history is therefore one
typed action out of a fixed set of nine, which is what makes "Run A took 5
experiments, Run B took 2" a comparison rather than an anecdote.

**Reranking, so a cold start is really cold.** Cosine similarity over 20 stored
lessons will always return the "closest" 5, even when none is relevant. The
`/rerank` pass applies a relevance floor, so early iterations correctly retrieve
*nothing*. Without it, Run A would also look like it had memory, and the
comparison collapses.

**Withdrawing tools, not just warning about them.** Retrieved lessons marked
`avoid` do two things: they appear in the prompt, and their action is removed
from the `tools` array for that call. Prompt text can be ignored by a model. A
tool that is not in the request cannot be called. This is the mechanism by which
persistent memory actually changes behaviour rather than decorating it.

---

## Setup

### 1. Credentials

```bash
cp .env.example .env
```

Fill in `FIREWORKS_API_KEY`. The hackathon code is `MONGODB813` ($50 credit,
deadline 10/1). Keys come from <https://app.fireworks.ai/settings/users/api-keys>.

`FIREWORKS_BASE_URL` defaults to `https://api.fireworks.ai/inference/v1` and only
needs setting if you are proxying.

### 2. Dependencies

`pcb_agent.fireworks` has exactly one hard dependency: **httpx**. Two optional
ones improve it:

| Package | Without it |
|---|---|
| `jsonschema` | a built-in validator covers the subset these schemas use |
| `langchain-fireworks` | use `chat_model_via_openai()`, which needs only `langchain-openai` |
| `python-dotenv` | export the variables yourself |

### 3. Verify

```bash
python scripts/fireworks_smoke.py
```

```
base_url: https://api.fireworks.ai/inference/v1
key:      fw_3a...9c1d

  Model registry vs live catalogue         ok   (0.6s)
  Chat completions                         ok   (0.9s)
  Streaming (SSE)                          ok   (1.1s)
  Structured output (json_schema)          ok   (2.4s)
  Tool calling (experiment allowlist)      ok   (1.8s)
  Grammar mode (BNF)                       ok   (0.5s)
  Embeddings for Atlas Vector Search       ok   (0.7s)
  Reranking                                ok   (0.6s)
  Vision (PCB crop inspection)             ok   (1.3s)
```

Run `--only models` for a single-call registry check, or `--only chat,tools` to
narrow it. Exit code is non-zero if anything failed.

---

## The model registry

**The Fireworks serverless catalogue rotates.** Models get promoted, demoted to
dedicated-only, and retired. A hardcoded model id that works today can return
404 next month. This is the single most common way this integration breaks, so
it is handled deliberately.

Nothing in the codebase names a model. Code asks for a **role**:

| Role | Default (verified Aug 2026) | Used by |
|---|---|---|
| `SCIENTIST` | `kimi-k2-instruct-0905` | diagnosis, experiment proposal |
| `CRITIC` | `deepseek-v3p1` | outcome judgement, lesson writing |
| `REASONER` | `kimi-k2-thinking` | escalation on hard diagnoses |
| `FAST` | `llama-v3p3-70b-instruct` | summaries, failure signatures |
| `VISION` | `qwen3-vl-235b-a22b-instruct` | misclassified crop inspection |
| `VISION_SMALL` | `qwen2p5-vl-32b-instruct` | one-off dataset descriptions |
| `EMBEDDING` | `qwen3-embedding-8b` | Atlas Vector Search vectors |
| `RERANK` | `qwen3-reranker-8b` | retrieval precision pass |

All ids are under `accounts/fireworks/models/`.

Three layers of protection:

1. **Env override.** `FIREWORKS_MODEL_SCIENTIST=accounts/fireworks/models/glm-4p6`
   repoints a role with no code change.
2. **Validation.** `validate_registry(client.list_models())` compares the roster
   against the live catalogue and reports which roles are dead, auto-substituting
   from a same-capability fallback list.
3. **Alternate spelling retry.** Fireworks accepts both
   `accounts/fireworks/models/x` and `fireworks/x` depending on the endpoint. On
   a 404 the client retries once with the other spelling before failing.

```bash
python scripts/fireworks_smoke.py --models
```

```
Fireworks catalogue: 148 models visible
  ok       critic        accounts/fireworks/models/deepseek-v3p1
  ok       embedding     accounts/fireworks/models/qwen3-embedding-8b
  fallback scientist     .../kimi-k2-instruct-0905 -> .../deepseek-v3p1
  DEAD     vision        accounts/fireworks/models/qwen3-vl-235b-a22b-instruct
```

### The embedding model is pinned

`Role.EMBEDDING` is in `models.PINNED_ROLES` and is **never** auto-substituted.
Changing the embedding model or its `dimensions` invalidates every vector already
in Atlas. The registry reports it as DEAD and stops rather than silently
poisoning retrieval. Repointing it is a deliberate act that requires re-embedding
the `lessons` collection and rebuilding the index.

---

## Capability guide

### 1. Chat completions

`FireworksClient` is a hand-rolled httpx client rather than the OpenAI SDK,
because Fireworks exposes parameters the OpenAI-typed clients do not model
(`response_format` grammar mode, `perf_metrics_in_response`,
`prompt_cache_isolation_key`, `reasoning_effort`, `return_token_ids`) and the
`/rerank` endpoint has no OpenAI equivalent at all. Routing chat, embeddings,
and rerank through one client also means retry, telemetry, and model fallback
behave identically everywhere.

```python
payload = client.chat(
    [{"role": "user", "content": "..."}],
    role=Role.SCIENTIST,
    temperature=0.2,
    max_tokens=2048,
    seed=42,                   # reproducible runs
    purpose="diagnose",        # shows up in the usage ledger
)
```

`purpose` is not sent to Fireworks. It labels the call in `client.ledger`, which
is how you find out that critique is costing three times what diagnosis costs.

### 2. Structured outputs (JSON schema)

The Fireworks docs flag two footguns, and `complete_json` handles both:

- The schema must appear **in the prompt and in `response_format`**. The model is
  not shown the schema automatically, and without it you get valid-but-wrong JSON.
- Without an explicit instruction to emit JSON, "the model may generate
  whitespace indefinitely until hitting token limits".

```python
from pcb_agent.fireworks import complete_json
from pcb_agent.fireworks.schemas import DIAGNOSIS_SCHEMA

result = complete_json(
    client,
    name="Diagnosis",
    schema=DIAGNOSIS_SCHEMA,
    messages=[{"role": "system", "content": SCIENTIST_SYSTEM},
              {"role": "user", "content": prompt}],
    role=Role.SCIENTIST,
)
```

On a validation failure it re-asks with the specific errors appended, up to
`FIREWORKS_STRUCTURED_MAX_ATTEMPTS` (default 3). On `finish_reason="length"` it
fails immediately with a message telling you to raise `max_tokens`, because
retrying truncated JSON at the same budget just burns credit.

Shipped schemas, all with `additionalProperties: false` so the constrained
decoder cannot invent fields:

| Schema | Purpose |
|---|---|
| `DIAGNOSIS_SCHEMA` | diagnosis, failure signature, hypothesis, confidence, ruled-out actions |
| `PROPOSAL_SCHEMA` | chosen action, parameters, expected effect, predicted delta |
| `CRITIQUE_SCHEMA` | verdict, explanation, and the MongoDB lesson document |
| `VISION_REPORT_SCHEMA` | visual observations, defect scale, image quality issues |
| `SECOND_OPINION_SCHEMA` | the OpenRouter judge's output, so both providers answer in the same shape |
| `FAILURE_SIGNATURE_SCHEMA` | cheap embeddable summary, generated by the FAST model |

`SECOND_OPINION_SCHEMA` living here is deliberate. Person 3 sends it to
OpenRouter, but defining it alongside the Fireworks schemas is what makes the two
providers' answers directly comparable instead of two differently-shaped blobs.

### 3. Tool calling (the experiment allowlist)

The brief: *"The model should select from executable tools rather than merely
suggesting changes in text."* All nine actions are real tools with typed
parameter schemas.

```
change_model            change_learning_rate    change_batch_size
change_image_size       change_augmentation     change_sampler
change_class_weights    change_confidence_threshold    retrain
```

```python
proposal = propose_experiment(client, state, diagnosis, memories=memories)
# {'next_action': 'change_sampler',
#  'parameters': {'strategy': 'weighted'},
#  'rationale': 'open_circuit is 3% of samples and recall is 0.32',
#  'expected_effect': '...', 'confidence': 0.78,
#  'second_opinion_requested': False}

new_config = apply_experiment(current_config, proposal)
```

Every tool carries a required `reason` argument. It costs a few tokens and gives
the critic and the stored lesson the model's stated intent. Without it a stored
experiment is just a config diff, and diffs do not transfer between datasets.

Validation is strict and happens before anything is executed:

| Rejected | Why |
|---|---|
| prose instead of a tool call | `tool_choice="required"` plus an explicit check |
| an unknown tool | not in the allowlist |
| a withdrawn tool | excluded for this step by retrieved memory |
| `learning_rate: 42.0` | violates the parameter schema's `maximum` |
| `change_class_weights(mode="manual")` with no weights | cross-field rule |
| unparseable argument JSON | caught before it reaches the trainer |

A bad argument fails here, in milliseconds, instead of 40 minutes into a training
job.

**Narrowing the action space.** This is the memory mechanism:

```python
withdrawn = actions_to_withdraw(memories, failure_mode)   # from `avoid` lessons
allowed   = [a for a in ACTIONS if a not in withdrawn]
proposal  = propose_experiment(client, state, diagnosis, allowed_actions=allowed)
```

`ActionHistory.allowed_actions` never narrows below three tools, so the model
always has a real choice and cannot be cornered into a forced action.

### 4. Embeddings for Atlas Vector Search

The scaffold shipped Voyage placeholders. Fireworks serves Voyage models *and*
Qwen3 embeddings through the same key and base URL, so the memory layer can run
on the Fireworks credit instead of a second vendor and a second key.

```python
from pcb_agent.fireworks import FireworksEmbedder

embedder = FireworksEmbedder(client)          # dimensions=1024 by default
vectors  = embedder.embed_documents([...])    # batched, cached, order-preserving
query    = embedder.embed_query(failure_summary)
```

`qwen3-embedding-8b` supports Matryoshka truncation, so `dimensions=1024` keeps
the Atlas index small without retraining anything.

The index definition comes from the embedder, so the two cannot drift apart:

```python
embedder.atlas_index_definition()
# {'fields': [{'type': 'vector', 'path': 'embedding',
#              'numDimensions': 1024, 'similarity': 'cosine'},
#             {'type': 'filter', 'path': 'failure_mode'},
#             {'type': 'filter', 'path': 'reusable'},
#             {'type': 'filter', 'path': 'avoid'}]}
```

**Query and document text must match in register.** The document side embeds
`critic.embedding_text(...)`, built from the failure summary plus the failure
mode and affected classes. The query side embeds
`diagnosis.failure_signature.summary`. Both describe the *problem*. If the
document side embedded the intervention instead, similarity search would start
matching on the wording of fixes rather than the shape of failures.

Every stored vector carries `embedding_model`, `embedding_dimensions`, and
`embedding_provider`, so a later model change is detectable rather than silently
corrupting retrieval.

### 5. Reranking

Two-stage retrieval:

```
Atlas Vector Search  ->  top 20 by cosine     (recall, cheap, approximate)
Fireworks /rerank    ->  top 5 by relevance   (precision, cross-encoder)
relevance floor 0.35 ->  possibly 0 lessons   (honest cold start)
```

```python
candidates = store.retrieve_similar_lessons(failure_summary, k=12)
memories   = retrieve_memories(client, failure_summary, candidates, top_k=4)
```

The floor matters more than the ordering, for the reason given in
[Why each capability is used](#why-each-capability-is-used).

If the rerank model is unavailable, `retrieve_memories` logs a warning and falls
back to vector order. Fewer, worse memories beats a crashed run.

### 6. Vision: inspecting misclassified crops

This is the one place the agent gets information that is not already in the
metrics. A confusion matrix says `missing_hole` recall is 0.41. It cannot say
"the defects are 12 pixels wide at this input resolution" or "half of these crops
are out of focus". Those point at completely different actions
(`change_image_size` versus a data-quality flag), and a text-only scientist
cannot tell them apart.

```python
from pcb_agent.fireworks.vision import collect_worst_crops, inspect_misclassified

paths, captions = collect_worst_crops(predictions, limit=12)
report = inspect_misclassified(client, paths, labels=captions,
                               metrics_context=metrics)
diagnosis = diagnose(client, state, memories, vision_report=report)
```

`collect_worst_crops` ranks by *model confidence in the wrong answer*, not by
being borderline. Those are the systematic errors worth looking at.

Documented limits are enforced client-side with actionable errors: 30 images per
request, 10MB total base64 payload, and PNG/JPG/GIF/BMP/TIFF/PPM only.
`inspect_misclassified` defaults to 12 images, since more mostly adds latency.

### 7. Streaming

```python
for piece in client.stream_text(messages, role=Role.SCIENTIST):
    print(piece, end="", flush=True)
```

`stream_chat` yields raw SSE chunks if you want to watch tool-call arguments
arrive incrementally. `stream_options: {include_usage: true}` is set by default,
so streamed calls still land in the usage ledger. Malformed SSE frames are
skipped rather than killing the stream.

### 8. Grammar mode

For outputs that are not JSON, a BNF grammar is cheaper and more reliable than a
one-field JSON object:

```python
from pcb_agent.fireworks.structured import VERDICT_GRAMMAR, grammar_response_format

payload = client.chat(messages, response_format=grammar_response_format(VERDICT_GRAMMAR))
# always exactly one of: improved | regressed | no_change | mixed
```

The agent itself computes verdicts arithmetically (see
[Reliability](#reliability)), so this is used for cheap classification side-tasks
rather than on the critical path.

### 9. Prompt caching

Prompt caching is on by default for all Fireworks models and requires no
configuration. It works on exact prefix matches, so the code is structured to
keep the cacheable part stable: the system prompt and the nine tool definitions
are byte-identical on every call, and only the metrics and retrieved memories
vary. Across a 5-experiment run that is a large, unchanging prefix.

`FIREWORKS_PROMPT_CACHE_ISOLATION_KEY` sets `prompt_cache_isolation_key` on
requests and the `x-prompt-cache-isolation-key` header. Set it per run
(`run_a`, `run_b`) when you want the cold-start and experienced runs to be
genuinely independent.

Note: cache-hit metrics are returned only for dedicated deployments, via the
`fireworks-cached-prompt-tokens` header. Serverless does not report them, so
`cached_prompt_tokens` will be `None` on serverless. The client reads the header
when present.

### 10. Reasoning effort

```bash
FIREWORKS_REASONING_EFFORT=high
```

Applies to models that support extended thinking (`low`, `medium`, `high`,
`max`, `none`). The intended pattern is escalation: run the normal loop at the
default, and re-diagnose with `Role.REASONER` plus high effort when the scientist
reports low confidence or two experiments have failed in a row. That escalation
policy is evaluated in `scientist._wants_second_opinion` and surfaced as
`proposal["second_opinion_requested"]`, so the LangGraph node is a one-line
check.

---

## Cost and token accounting

$50 of credit is a hard budget, and the headline claim ("the experienced agent
needed fewer experiments") invites the question of what it cost. Every call is
recorded.

```python
client.ledger.totals()
# {'run_id': 'run_a', 'calls': 23, 'failed_calls': 0,
#  'prompt_tokens': 84102, 'completion_tokens': 9871, 'total_tokens': 93973,
#  'cost_usd': None, 'cost_coverage': '0/23',
#  'latency_s_total': 61.4, 'retries': 1}

client.ledger.by_purpose()
# {'diagnose': {'calls': 5, 'total_tokens': 41203, ...},
#  'propose_experiment': {'calls': 5, 'total_tokens': 28110, ...},
#  'critique': {'calls': 5, 'total_tokens': 19_...}, ...}
```

**Prices are not hardcoded.** Fireworks per-token pricing varies by model and
changes, and a stale number in source is worse than no number. Point
`FIREWORKS_PRICE_TABLE` at a JSON file and `cost_usd` populates:

```json
{
  "accounts/fireworks/models/kimi-k2-instruct-0905": {"input": 0.60, "output": 2.50},
  "accounts/fireworks/models/deepseek-v3p1":         {"input": 0.56, "output": 1.68}
}
```

Values are USD per 1M tokens; take current figures from
<https://fireworks.ai/pricing>. `cost_coverage` reports how many calls the table
actually covered, so partial tables cannot masquerade as complete ones.

`ledger.to_documents()` produces documents for a MongoDB `fireworks_calls`
collection if Person 1 wants per-run cost queryable alongside the experiments.

### Keeping the bill down

- Role split. Cheap work (summaries, failure signatures) goes to `Role.FAST`
  rather than the scientist model.
- Compact history. `_render_history` sends the last 8 experiments as one line
  each, not full records. By iteration 4 the difference is most of the context.
- Vectors are stripped from retrieved lessons before they reach a prompt.
- Embedding cache. The same failure signature is embedded once per process.
- `_top_deltas` caps the critic's metric payload at 12 entries, ranked by
  absolute change, so datasets with many classes do not blow up the prompt.

---

## Reliability

**Retries.** 429 and 5xx and timeouts retry with exponential backoff plus jitter,
honouring `Retry-After`. 4xx does not retry, because the request will not become
valid on its own. Configure with `FIREWORKS_MAX_RETRIES`, `FIREWORKS_BACKOFF_BASE`,
`FIREWORKS_BACKOFF_MAX`.

**Typed errors,** so the orchestrator can branch on recovery rather than parse
strings:

```
FireworksError
├── FireworksConfigError            missing key, bad base URL
├── FireworksAuthError              401/403
├── FireworksNotFoundError          404, usually a retired model id
├── FireworksRateLimitError         429, carries retry_after
├── FireworksServerError            5xx
├── FireworksTimeoutError           network/read timeout
├── FireworksBadRequestError        other 4xx
├── FireworksStructuredOutputError  schema validation failed after repairs
└── FireworksToolChoiceError        no tool call, unknown tool, bad arguments
```

`is_retryable(exc)` covers the retryable three.

**Hallucinated citations are dropped.** The scientist is asked which lesson ids
it used. Any id not in the retrieved set is removed before the diagnosis reaches
MongoDB.

**Verdicts are arithmetic, not opinion.** The critic writes a verdict, and then
`critique_result` overwrites it with the measured delta against a noise
threshold (default 0.01 macro F1). A generous critic cannot talk a run past its
stop condition, and changes within noise are marked `reusable: false` so they are
recorded but never retrieved as guidance.

**Forecast calibration is tracked.** The scientist's `expected_delta_macro_f1` is
compared against the actual delta on every experiment. Improving calibration
across a run is a second, independent piece of evidence that the agent got better.

---

## LangChain and LangGraph

The agent itself does not require LangChain. `scientist.py` and `critic.py` use
`FireworksClient` directly so that structured output, tool validation, retries,
and token accounting behave identically whether or not LangChain is installed.

For Person 3's graph nodes, `langchain_adapters.py` offers two routes:

```python
from pcb_agent.fireworks.langchain_adapters import (
    chat_model, chat_model_via_openai, embeddings, bind_experiment_tools,
    langsmith_metadata,
)

llm = chat_model(Role.SCIENTIST)              # needs langchain-fireworks
llm = chat_model_via_openai(Role.SCIENTIST)   # needs only langchain-openai
llm = bind_experiment_tools(llm)              # attaches the allowlist
```

Fireworks is OpenAI-compatible, so `chat_model_via_openai` is just `ChatOpenAI`
pointed at the Fireworks base URL, with Fireworks-only parameters passed through
`model_kwargs`. It needs no new dependency.

For Atlas:

```python
from langchain_mongodb import MongoDBAtlasVectorSearch

store = MongoDBAtlasVectorSearch(
    collection=db["lessons"],
    embedding=embeddings(),          # this package's client: retries, cache, ledger
    index_name="lessons_vector_index",
)
```

`langsmith_metadata()` returns the model roster for trace tagging, which is what
lets you tell a Run A trace from a Run B trace afterwards.

---

## Mapping to the 3-person plan

The three frozen interfaces from `PCB_AGENT_3_PERSON_BACKEND_PLAN.md` are
implemented exactly as specified, with the client passed as the first argument:

| Plan signature | Implementation |
|---|---|
| `diagnose(state, memories) -> dict` | `scientist.diagnose(client, state, memories)` |
| `propose_experiment(state, diagnosis) -> dict` | `scientist.propose_experiment(client, state, diagnosis)` |
| `critique_result(before, after, action) -> dict` | `critic.critique_result(client, before, after, action)` |
| `apply_experiment(current_config, proposal) -> dict` | `tools.apply_experiment(config, proposal)` |
| `second_opinion(context) -> dict` | schema only (`SECOND_OPINION_SCHEMA`); Person 3 owns the OpenRouter call |

Person 1's memory interface is stubbed in `pcb_agent/demo/memory_store.py` with
the same signatures (`store_experiment`, `store_lesson`,
`retrieve_similar_lessons`, `get_run_history`), so the Fireworks side is testable
before Atlas exists and swaps over without changes upstream.

Person 2's `train_model` / `evaluate_model` are stubbed in
`pcb_agent/ml/simulator.py`. That module is deliberately a placeholder, but it is
not a random number generator: each simulated task has hidden bottlenecks (class
imbalance, defect scale, capacity, calibration), and macro F1 responds only to
interventions that address a real one. An agent that reasons correctly
measurably outperforms one that guesses, which is what makes the demo worth
running. `TASK_A` starts at 0.578 macro F1, matching the brief's 0.58 baseline.

### The killer evaluation

```bash
python -m pcb_agent.demo.fireworks_loop --run a --budget 5 --fresh   # cold start
python -m pcb_agent.demo.fireworks_loop --run b --budget 5           # experienced
```

Run A writes lessons to `.omc/fireworks_demo_lessons.json`. Run B reads them and
faces a related task with a different class mix. `--run both` prints the
comparison, and reports honestly when the result is inconclusive rather than
claiming a win.

---

## Module reference

```
pcb_agent/fireworks/
├── __init__.py            public surface, one import for everything
├── config.py              FireworksSettings, all env parsing
├── models.py              Role enum, registry, validate_registry
├── client.py              HTTP: chat, stream, embed, rerank, list_models
├── errors.py              typed exception hierarchy
├── telemetry.py           UsageLedger, CallRecord, price table
├── schemas.py             JSON Schemas for every structured response
├── structured.py          complete_json, repair loop, validators
├── tools.py               the 9-action allowlist, parsing, application
├── scientist.py           diagnose, propose_experiment
├── critic.py              critique_result, lesson documents
├── embeddings.py          FireworksEmbedder, Atlas index definition
├── rerank.py              two-stage retrieval, relevance floor
├── vision.py              PCB crop inspection
└── langchain_adapters.py  LangChain / LangGraph wiring

pcb_agent/ml/simulator.py       stand-in trainer with real bottlenecks
pcb_agent/demo/memory_store.py  stand-in for Person 1's Atlas layer
pcb_agent/demo/fireworks_loop.py  the full loop, runnable today
scripts/fireworks_smoke.py      live capability check
```

---

## Testing

```bash
pytest tests/ -q          # 157 tests, no API key, no network
```

Every test runs offline against `httpx.MockTransport`, including a full
end-to-end loop with a scripted Fireworks. Coverage includes request assembly,
error mapping and retry counts, the alternate-model-id retry, SSE parsing,
structured-output repair, tool-argument rejection, the relevance floor, embedding
batching and caching, and both validator implementations.

The live smoke test is the only thing that needs a key:

```bash
python scripts/fireworks_smoke.py
```

---

## Troubleshooting

**`FIREWORKS_API_KEY is not set`** - copy `.env.example` to `.env` and fill it
in. The demo and smoke script load `.env` automatically when `python-dotenv` is
installed; otherwise export it yourself.

**404, "Fireworks has no model X"** - the serverless catalogue rotated. Run
`python scripts/fireworks_smoke.py --models`, then set the `FIREWORKS_MODEL_*`
override for the dead role. Browse <https://app.fireworks.ai/models> for a live
replacement in the same capability class.

**`FireworksToolChoiceError: model returned no tool call`** - the model answered
in prose. `propose_experiment` already sets `tool_choice="required"`; if you are
calling `client.chat` directly, set it. If it persists, the model may not support
tool calling: check `supportsTools` on its model page.

**`FireworksStructuredOutputError` after N attempts** - read
`exc.raw` to see what the model actually produced. Usually the schema has an
overly strict `enum` the model keeps missing, or `max_tokens` is too low.

**"hit the token limit and was truncated"** - raise `max_tokens`. Retrying at the
same budget cannot succeed, so this fails immediately by design.

**Rerank returns nothing** - working as intended when no stored lesson is
relevant. If it is over-filtering, lower `relevance_floor` from the default 0.35,
but keep it above zero.

**Atlas returns no vector results** - `numDimensions` on the index must equal
`FIREWORKS_EMBEDDING_DIMENSIONS`. Get the definition from
`embedder.atlas_index_definition()`. If you changed the embedding model, every
existing vector is invalid and the collection must be re-embedded.

**Every request is slow** - set `FIREWORKS_PERF_METRICS=true` for a server-side
timing breakdown in `CallRecord.perf_metrics`, and check
`ledger.totals()["retries"]` for silent rate limiting.

---

## Not yet integrated

Deliberate omissions, with the reason:

| Capability | Why not |
|---|---|
| Fine-tuning (SFT / DPO / RFT) | The plan's "Do Not Build Yet" list. Worth revisiting only after enough lessons accumulate to make a scientist-policy fine-tune meaningful. |
| Batch API | Discounted async processing suits offline evaluation sweeps, not an interactive loop where each step depends on the last. |
| Responses API | The Chat Completions surface covers everything needed here. |
| Anthropic-compatible endpoint | Redundant with the OpenAI-compatible path already in use. |
| Async client | The loop is inherently sequential. The retry, telemetry, and structured-output code is transport-agnostic, so adding `httpx.AsyncClient` is mechanical if the graph ever fans out. |
| Dedicated deployments | Serverless is sufficient at hackathon volume, and dedicated would change the cost model entirely. |
| Video and audio inputs | No use case in a still-image defect dataset. |

---

## References

- [Fireworks API reference](https://docs.fireworks.ai/api-reference/introduction)
- [Chat completions](https://docs.fireworks.ai/api-reference/post-chatcompletions)
- [Structured outputs](https://docs.fireworks.ai/structured-responses/structured-response-formatting)
- [Tool calling](https://docs.fireworks.ai/guides/function-calling)
- [Embeddings and reranking](https://docs.fireworks.ai/guides/querying-embeddings-models)
- [Vision models](https://docs.fireworks.ai/guides/querying-vision-language-models)
- [Prompt caching](https://docs.fireworks.ai/guides/prompt-caching)
- [Model library](https://app.fireworks.ai/models)
- [Pricing](https://fireworks.ai/pricing)
