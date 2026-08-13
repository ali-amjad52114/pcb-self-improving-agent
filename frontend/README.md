# Switchback AI

**AutoML that remembers how it learned.**

Switchback AI is a self-improving experiment agent built for the MongoDB Persistent
Context Sprint. It records model experiments, outcomes, critiques, and reusable
lessons so a new but structurally similar ML problem does not begin from zero.

## Demo story

1. A cold PCB classifier begins at 55% macro F1 with weak minority-defect recall.
2. A Fireworks-powered critic diagnoses each failure and chooses a real training intervention.
3. Weighted sampling, localized crops, higher resolution, and threshold calibration lift macro F1 to 84%.
4. MongoDB stores the problem fingerprint, intervention chain, outcome, and verified lesson.
5. On a related PCB batch, Atlas Vector Search and Voyage AI retrieve that experience.
6. The experienced agent reaches the same 84% target in 2 experiments instead of 5.

## Current integration status

The frontend is a transparent guided mock and is ready to consume backend events.
Its components map to dataset profiling, experiment updates, critic decisions,
MongoDB memory retrieval, transfer runs, and final verification. Until the API
contract is connected, no simulated event is presented as a live backend result.

## Partner architecture

- **MongoDB Atlas:** persistent experiment memory, state, checkpoints, and Vector Search
- **Voyage AI:** dataset and experience embeddings
- **Fireworks AI:** experiment actor and outcome critic
- **OpenRouter:** independent lesson verification
- **LangGraph:** experiment workflow orchestration and resumption
- **LangSmith:** traces, comparisons, and evaluation

## Run locally

Requires Node.js 22.13 or newer.

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Validate

```bash
npm run build
```

The current interface includes a complete mock-data demo path so frontend and
agent development can proceed independently. Replace the in-component demo
records with the agent API response when the backend contract is ready.
