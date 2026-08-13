# PCB Self-Improving Agent — Person 3 Orchestrator

LangGraph orchestration for a self-improving PCB defect classification agent.

Person 3 owns: graph control flow, MongoDB LangGraph checkpointing, OpenRouter
independent evaluator, teammate adapters, fake adapters, and CLI runner.

## Architecture

```text
                   ┌──────────────────────┐
                   │      LangGraph       │
                   │     Orchestrator     │
                   └──────────┬───────────┘
                              │
               ┌──────────────┼──────────────┐
               │              │              │
               ↓              ↓              ↓
          MongoDB         Fireworks      OpenRouter
          Memory          Scientist       Judge
               ↑              │
               └──── outcome ─┘

          MongoDB also stores
          LangGraph checkpoints
```

**Long-term memory ≠ LangGraph checkpoint state**

| Concern | What it stores | Owner |
| --- | --- | --- |
| Long-term experience | experiments, lessons, vector retrieval | Person 1 |
| LangGraph checkpoint | workflow/node state, crash recovery, resume | Person 3 |

## Graph topology

```text
START → train_baseline → evaluate_baseline → retrieve_memory → diagnose
  → propose_experiment → needs_proposal_judge?
        ├─ NO ──────────────────────────────→ run_experiment
        └─ YES → judge_proposal → reconsider?
                    ├─ YES (once) → propose_experiment
                    └─ NO → run_experiment
  → evaluate_result → critique_result → final_validation?
        ├─ YES → judge_lesson → store_experience
        └─ NO ───────────────→ store_experience
  → should_stop? → END | retrieve_memory
```

## Setup (Windows PowerShell)

```powershell
cd C:\Users\<USERNAME>\pcb-self-improving-agent

uv sync

Copy-Item .env.example .env
```

## Run locally (fake mode)

```powershell
uv run python -m pcb_agent.runner start --run-id demo_001
```

Optional baseline config:

```powershell
uv run python -m pcb_agent.runner start `
  --run-id run_cold_001 `
  --config .\configs\baseline.json
```

## Resume

```powershell
uv run python -m pcb_agent.runner resume --run-id demo_001
```

## Cold vs experienced demo (fake)

```powershell
# Cold start — inefficient exploration, stores lessons to data\fake_memory.json
uv run python -m pcb_agent.runner start --run-id run_cold_001

# Experienced — retrieves prior lessons and changes experiment order
uv run python -m pcb_agent.runner start --run-id run_warm_001
```

Fake memory file persistence is labeled:

```text
[FALLBACK ONLY — FINAL DEMO SHOULD USE MONGODB]
```

## Integrated mode (Persons 1 + 2)

Set in `.env`:

```env
AGENT_MODE=integrated

MEMORY_MODULE=memory.service
ML_MODULE=ml.pipeline
REASONING_MODULE=scientist.agent

CHECKPOINTER_BACKEND=mongodb
MONGODB_URI=<from Person 1 / Atlas>
MONGODB_DB_NAME=pcb_self_improving_agent

OPENROUTER_ENABLED=true
OPENROUTER_API_KEY=<provided>
OPENROUTER_MODEL=<provided>
```

Then:

```powershell
uv run python -m pcb_agent.runner start --run-id integrated_001
```

`graph.py`, `routing.py`, and `state.py` do not change for teammate integration.

## Tests

```powershell
uv run pytest -q
```

All unit tests inject `FakeJudgeAdapter` and make **zero** live OpenRouter calls.

## OpenRouter policy

Judge is called only when:

1. diagnosis confidence < 0.65
2. proposal confidence < 0.65
3. two consecutive failed experiments
4. proposal `next_action == change_model`
5. final lesson validation at run end (when enabled)

Otherwise the console shows:

```text
[JUDGE] skipped — scientist confidence sufficient
```

Failures (timeout, 401/402/429/5xx, invalid JSON) return a safe fallback and the loop continues.

## API / external service risk register

| Service | Risk | Required behavior |
| --- | --- | --- |
| MongoDB Atlas | URI/network/auth failure | Clear startup failure if Mongo checkpointer explicitly required |
| Person 1 memory module | Not finished yet | FakeMemory fallback |
| Fireworks / Person 2 reasoning | Rate limit or unfinished integration | FakeReasoning for local orchestration testing |
| Person 2 training | Slow training | FakeML for orchestration tests |
| OpenRouter | 402/429/timeouts/structured-output issues | Conditional calls, max retries, safe skip/fallback |
| LangGraph MongoDB checkpointer | Package/API mismatch | Inspect installed API; fail clearly if Mongo required |

## Ownership

| Area | Status |
| --- | --- |
| LangGraph orchestrator | Person 3 |
| MongoDB vector memory | Person 1 |
| Fireworks scientist + real ML train | Person 2 |
| OpenRouter independent judge | Person 3 |
