# PCB Self-Improving Agent

Backend-only agent for the [MongoDB Persistent Context Sprint](https://cerebralvalley.ai/e/persistent-context-sprint-hackathon) at `.local` Build Fest (Aug 13, 2026).

The classifier is the task. The product is an agent that gets better at improving PCB defect classifiers over time:

**Train → Evaluate → Diagnose → Retrieve past experience → Choose next experiment → Retrain → Store lesson**

MongoDB Atlas holds experiment history, failure memory, and vector-search retrieval so the agent does not cold-start.

See [`PCB_AGENT_3_PERSON_BACKEND_PLAN.md`](./PCB_AGENT_3_PERSON_BACKEND_PLAN.md) for the 3-person backend split.

## Person 1 — MongoDB memory (this repo)

Atlas is the agent's long-term experience, not a log. The memory layer uses:

- JSON Schema validation on `experiments` and `lessons`
- Voyage `voyage-3-large` embeddings (1024-d) + Vector Search
- Atlas Search lexical index + `$rankFusion` hybrid retrieval
- Transactions when committing an experiment and lesson together
- Computed `run_summaries` and aggregations (`get_run_history`, `best_interventions`)
- Change streams on `lessons`
- LangGraph MongoDB checkpointer helper for Person 3

```python
from pcb_memory import AgentMemory

memory = AgentMemory()
memory.bootstrap()
memory.store_lesson({...})
memory.retrieve_similar_lessons("Small open circuits have low recall and class imbalance")
memory.get_run_history("run_cold_start_demo")
memory.checkpointer()  # Person 3
```

```bash
python -m scripts.bootstrap_memory   # needs MONGODB_URI + VOYAGE_API_KEY
python -m scripts.demo_retrieve
```

## Stack

- MongoDB Atlas sandbox (Vector Search + LangGraph checkpointer)
- LangChain / LangGraph
- Fireworks (code `MONGODB813`)
- OpenRouter, ElevenLabs, LangSmith (partner credits)

## Setup

```bash
cp .env.example .env
# fill MONGODB_URI and partner keys

python -m venv .venv
.\.venv\Scripts\activate   # Windows
pip install -r requirements.txt

npm install
```

Do not commit `.env`. The hackathon build must live in the Atlas Hackathon Sandbox to be eligible for finalists.
