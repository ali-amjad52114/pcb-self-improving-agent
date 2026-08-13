# PCB Self-Improving Agent

Backend-only agent for the [MongoDB Persistent Context Sprint](https://cerebralvalley.ai/e/persistent-context-sprint-hackathon) at `.local` Build Fest (Aug 13, 2026).

The classifier is the task. The product is an agent that gets better at improving PCB defect classifiers over time:

**Train → Evaluate → Diagnose → Retrieve past experience → Choose next experiment → Retrain → Store lesson**

MongoDB Atlas holds experiment history, failure memory, and vector-search retrieval so the agent does not cold-start.

See [`PCB_AGENT_3_PERSON_BACKEND_PLAN.md`](./PCB_AGENT_3_PERSON_BACKEND_PLAN.md) for the 3-person backend split.

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
