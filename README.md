# PCB Self-Improving Agent

Backend-only agent for the MongoDB Persistent Context Sprint. The classifier is
the task; the product is an agent that gets better at improving PCB defect
classifiers over time.

```text
Train → Evaluate → Retrieve Atlas memory → Diagnose → Choose experiment
      → Retrain → Critique → Store lesson → Repeat
```

## Integrated architecture

```text
                   LangGraph (Person 3)
                          │
             ┌────────────┼────────────┐
             ↓            ↓            ↓
       MongoDB Atlas   Fireworks    OpenRouter
        (Person 1)     (Person 2)   (Person 3)
             ↑            │
             └── experiment outcome ┘
```

MongoDB Atlas is the agent's long-term experience, not just a log. It stores
validated experiments and lessons, Voyage embeddings, hybrid Vector Search,
computed run evidence, and LangGraph checkpoints.

Long-term experience and graph checkpoints are separate concerns:

| Concern | Data | Owner |
| --- | --- | --- |
| Long-term experience | experiments, lessons, retrieval, run analytics | Person 1 |
| Workflow checkpoint | graph/node state and crash recovery | Person 3 |

## Person 1 — MongoDB memory

```python
from pcb_memory import AgentMemory

memory = AgentMemory()
memory.bootstrap()
memory.retrieve_similar_lessons("Small open circuits have low recall")
memory.commit_experience(experiment, lesson)
memory.get_run_history("run_001")
memory.compare_runs(["run_cold", "run_warm"])
memory.checkpointer()
```

Live Atlas sponsor demo:

```powershell
python -m scripts.demo_atlas_memory
```

See [PERSON_1_MONGODB_HANDOFF.md](./PERSON_1_MONGODB_HANDOFF.md) for the exact
contract and judge-facing story.

## Person 3 — LangGraph orchestrator

Graph topology:

```text
START → train_baseline → evaluate_baseline → retrieve_memory → diagnose
  → propose_experiment → optional proposal judge → run_experiment
  → evaluate_result → critique_result → optional lesson judge
  → store_experience → stop or retrieve_memory
```

Run the orchestration locally with fake teammate adapters:

```powershell
python -m pcb_agent.runner start --run-id demo_001
python -m pcb_agent.runner resume --run-id demo_001
```

Optional baseline configuration:

```powershell
python -m pcb_agent.runner start --run-id run_cold_001 --config .\configs\baseline.json
```

## Integrated mode

Set these values in `.env` after Person 2 supplies their modules:

```env
AGENT_MODE=integrated
MEMORY_MODULE=pcb_memory.memory
ML_MODULE=<person-2-ml-module>
REASONING_MODULE=<person-2-reasoning-module>

CHECKPOINTER_BACKEND=mongodb
MONGODB_DB=persistent_context
MONGODB_DB_NAME=persistent_context

OPENROUTER_ENABLED=true
OPENROUTER_MODEL=<model-name>
```

Then run:

```powershell
python -m pcb_agent.runner start --run-id integrated_001
```

The graph, routing, and state do not change when switching from fakes to real
teammate modules.

## OpenRouter policy

Person 3 calls the independent judge only for low confidence, repeated failures,
model-family changes, or final lesson validation. Failures fall back safely so
the experiment loop can continue.

## Setup and tests

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
pip install -r requirements.txt
python -m unittest discover -s tests -v
pytest -q
```

The real build must use the Atlas Hackathon Sandbox. Never commit `.env`.
