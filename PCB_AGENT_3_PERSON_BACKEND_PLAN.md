# PCB Self-Improving Agent — 3-Person Backend Split

## Goal

Build a backend-only self-improving PCB defect classification agent.

Core loop:

**Train → Evaluate → Diagnose → Retrieve Past Experience → Choose Next Experiment → Retrain → Store Lesson**

The classifier is the task.  
The product is the **agent that becomes better at improving classifiers over time**.

---

# Team Split

## Person 1 — MongoDB / Memory Owner

### Owns
- MongoDB Atlas connection and collections
- Experiment history
- Failure / correction memory
- Vector Search
- Retrieval of similar past experiments
- LangGraph MongoDB checkpointer integration support
- Memory write/read API

### Collections

`experiments`
```json
{
  "experiment_id": "exp_012",
  "run_id": "run_3",
  "dataset_stats": {},
  "model_config": {},
  "metrics": {},
  "failure_signature": {},
  "intervention": {},
  "outcome_delta": {},
  "status": "complete"
}
```

`lessons`
```json
{
  "lesson_id": "lesson_22",
  "failure_summary": "Poor recall on minority open-circuit class",
  "intervention": "weighted sampler + tighter crops",
  "result": "+11% macro F1",
  "confidence": 0.91,
  "embedding": []
}
```

### Must expose

```python
store_experiment(experiment)
store_lesson(lesson)
retrieve_similar_lessons(failure_summary, k=5)
get_run_history(run_id)
```

### Definition of done

Given a new failure description:

> "Small open circuits have low recall and class imbalance"

MongoDB returns the most relevant previous lessons and their outcomes.

MongoDB is the agent's **long-term experience**, not just a log.

---

# Person 2 — Fireworks / ML Scientist Agent Owner

### Owns
- Fireworks API
- Scientist / actor agent
- Critic agent
- Structured outputs
- Experiment proposal logic
- Failure diagnosis
- Lesson generation
- Actual ML training/evaluation functions

### Scientist input

```json
{
  "dataset_summary": {},
  "current_model": {},
  "metrics": {},
  "confusion_matrix": {},
  "past_lessons": []
}
```

### Scientist output

```json
{
  "diagnosis": "Minority classes are underrepresented",
  "hypothesis": "Weighted sampling should improve recall",
  "next_action": "change_sampler",
  "parameters": {
    "strategy": "weighted"
  },
  "expected_effect": "Improve macro F1"
}
```

Fireworks supports tool/function calling, so the model should choose from a small allowlist of real experiment tools rather than returning free-form advice.

### Initial tool allowlist

```text
change_model()
change_learning_rate()
change_batch_size()
change_image_size()
change_augmentation()
change_sampler()
change_class_weights()
change_confidence_threshold()
retrain()
```

### Critic

After each experiment:

```text
Previous metrics
+ action taken
+ new metrics
→ determine whether the intervention helped
→ produce reusable lesson
```

### Definition of done

Starting from a weak baseline, the Fireworks agent can autonomously:

1. Read evaluation results
2. Diagnose a likely failure
3. Choose one allowed intervention
4. Trigger retraining
5. Evaluate whether the intervention worked
6. Produce a lesson for MongoDB

---

# Person 3 — LangGraph + OpenRouter / Orchestration Owner

## LangGraph owns the control loop

### Graph

```text
START
  ↓
BASELINE / TRAIN
  ↓
EVALUATE
  ↓
RETRIEVE_MEMORY
  ↓
DIAGNOSE
  ↓
CHOOSE_EXPERIMENT
  ↓
RUN_EXPERIMENT
  ↓
EVALUATE_RESULT
  ↓
STORE_EXPERIENCE
  ↓
TARGET REACHED?
   ↙       ↘
 YES       NO
  ↓         ↘
 END    RETRIEVE_MEMORY
```

### Owns
- LangGraph state definition
- Node wiring
- Retry / failure behavior
- Run IDs
- Experiment budget
- Stop conditions
- MongoDB checkpointer connection
- OpenRouter integration

LangGraph checkpoints state at graph steps, and MongoDB provides a native LangGraph checkpointer. This means a run can crash during experiment #4 and resume without restarting the learning process.

### Suggested state

```python
class AgentState:
    run_id: str
    iteration: int
    dataset_summary: dict
    current_config: dict
    current_metrics: dict
    confusion_matrix: dict
    retrieved_lessons: list
    diagnosis: dict
    proposed_experiment: dict
    experiment_history: list
    best_metrics: dict
```

## OpenRouter role

Do **not** duplicate Fireworks.

Use OpenRouter as an independent **second-opinion evaluator / judge**.

Example:

```text
Fireworks:
"I think class imbalance is the primary problem."

OpenRouter evaluator:
"Agree / disagree + confidence + alternative explanation."
```

Only call OpenRouter when:
- Fireworks confidence is low
- two experiments fail consecutively
- the agent wants to switch model families
- final lesson needs independent validation

This makes OpenRouter meaningful rather than decorative.

### Definition of done

One command can launch the entire autonomous loop and it continues until:

```text
target metric reached
OR
experiment budget exhausted
```

It must survive interruption through checkpointing.

---

# Shared Interfaces — Freeze These First

Nobody should directly depend on another person's internal code.

## Memory

```python
retrieve_similar_lessons(query: str, k: int) -> list
store_experiment(data: dict) -> str
store_lesson(data: dict) -> str
```

## ML

```python
train_model(config: dict) -> dict
evaluate_model(model_id: str) -> dict
apply_experiment(current_config: dict, proposal: dict) -> dict
```

## Reasoning

```python
diagnose(state: dict, memories: list) -> dict
propose_experiment(state: dict, diagnosis: dict) -> dict
critique_result(before: dict, after: dict, action: dict) -> dict
```

## Independent evaluator

```python
second_opinion(context: dict) -> dict
```

---

# Integration Rule

Each engineer owns their subsystem, but everything communicates using plain JSON/dicts.

```text
              ┌──────────────┐
              │   LangGraph  │
              │ Orchestrator │
              └──────┬───────┘
                     │
          ┌──────────┼───────────┐
          ↓          ↓           ↓
     MongoDB      Fireworks   OpenRouter
     Memory       Scientist    Evaluator
          ↑          │
          └──── outcome ────────┘
```

---

# First Integration Milestone

Do **not** wait until everyone finishes.

Within the first hour, make this fake-data flow work:

```text
LangGraph
→ asks MongoDB for past lessons
→ sends lessons + fake metrics to Fireworks
→ Fireworks proposes experiment
→ OpenRouter optionally evaluates proposal
→ LangGraph stores fake outcome in MongoDB
→ next iteration retrieves it
```

Once this loop works, replace fake pieces with the real PCB training pipeline.

---

# Minimum Winning Backend Demo

Run A — Cold Start:

```text
Baseline: 55% macro F1
Experiment 1: 61%
Experiment 2: 69%
Experiment 3: 78%
Experiment 4: 84%
```

MongoDB accumulates the successful and failed interventions.

Run B — Experienced Agent:

```text
New / altered PCB task
↓
MongoDB retrieves relevant previous lessons
↓
Fireworks skips known-bad experiments
↓
84% reached in 2 experiments instead of 4
```

The key metric is not only:

> **Classifier accuracy increased.**

It is:

> **The experienced agent reached the target with fewer experiments than the cold-start agent.**

That proves persistent context changed what the agent did next.

---

# Ownership Summary

| Engineer | Primary ownership | Secondary |
|---|---|---|
| **1** | MongoDB Atlas + Vector Search + memory schema | MongoDB LangGraph checkpointer support |
| **2** | Fireworks scientist + critic + ML experiment tools | Training/evaluation pipeline |
| **3** | LangGraph orchestration | OpenRouter independent evaluator |

---

# Do Not Build Yet

- Frontend
- Fancy dashboards
- Authentication
- Multiple datasets
- Fine-tuning Fireworks
- Huge model zoo
- Complicated microservices

First prove:

> **Past experience stored in MongoDB causes Fireworks to choose a better next experiment through the LangGraph loop.**

Everything else is optional.
