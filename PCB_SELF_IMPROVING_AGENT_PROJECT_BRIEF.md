# Coding Agent Project Brief --- Self-Improving PCB Inspection Agent

## Project

Build a backend-first autonomous ML agent that trains a PCB defect
classifier, evaluates its performance, diagnoses why it is failing,
chooses an experiment to improve it, retrains, and repeats until it
reaches a target performance or exhausts its experiment budget.

The key idea is **No Cold Start**: the agent must learn from previous
experiments. When it encounters a similar failure in a future run, it
should retrieve what previously worked or failed and use that experience
to make a better next decision.

## Core Loop

**Train → Evaluate → Retrieve Memory → Diagnose → Propose Experiment →
Execute → Evaluate Outcome → Store Lesson → Repeat**

The system should demonstrate:

1.  **Classifier improvement** --- model performance increases during a
    run.
2.  **Agent improvement** --- an experienced agent reaches the target in
    fewer experiments than a cold-start agent.

## MongoDB Atlas + LangGraph

Use MongoDB as the persistence and memory layer for LangGraph.

Store: - LangGraph checkpoints/state - Experiment configurations -
Dataset/model metrics - Confusion matrices - Failure signatures -
Actions/interventions attempted - Before/after performance - Successful
and failed experiments - Distilled lessons

Use MongoDB Vector Search to retrieve relevant previous experiences when
a new failure is diagnosed.

Example memory:

``` json
{
  "failure": "Low recall for small open-circuit defects with class imbalance",
  "action": "weighted sampling + tighter image crops",
  "before_f1": 0.67,
  "after_f1": 0.79,
  "delta": 0.12,
  "lesson": "Weighted sampling and localized crops helped small minority defects."
}
```

LangGraph should use MongoDB checkpointing so an interrupted experiment
loop can resume without losing state.

## Fireworks AI

Fireworks is the primary intelligence layer.

### Scientist / Actor

Given current metrics, confusion matrix, experiment history, and
retrieved MongoDB memories: - Diagnose likely reason for poor
performance - Form a hypothesis - Select the next experiment - Return
structured parameters for execution

### Critic

After each experiment: - Compare before/after metrics - Determine
whether the hypothesis was correct - Explain why the intervention helped
or failed - Generate a reusable lesson for MongoDB

The model should select from executable tools rather than merely
suggesting changes in text.

Possible actions:

``` text
change_model
change_learning_rate
change_batch_size
change_image_size
change_augmentation
change_sampler
change_class_weights
change_confidence_threshold
retrain
```

## OpenRouter

Use OpenRouter as an independent second-opinion evaluator, not as a
duplicate of Fireworks.

Invoke it when: - Fireworks has low confidence - Two experiments fail
consecutively - A major strategy/model-family change is proposed - A
learned lesson needs independent validation

It should return agreement/disagreement, confidence, and an alternative
diagnosis when appropriate.

## LangGraph Flow

``` text
START
  ↓
TRAIN
  ↓
EVALUATE
  ↓
RETRIEVE RELEVANT EXPERIENCE FROM MONGODB
  ↓
FIREWORKS DIAGNOSE
  ↓
OPTIONAL OPENROUTER SECOND OPINION
  ↓
CHOOSE EXPERIMENT
  ↓
EXECUTE EXPERIMENT
  ↓
RETRAIN
  ↓
EVALUATE
  ↓
FIREWORKS CRITIQUE
  ↓
STORE EXPERIMENT + LESSON IN MONGODB
  ↓
TARGET REACHED?
   ↙       ↘
 YES       NO
  ↓         ↓
 END      LOOP
```

## Most Important Behavior

Do **not** build a fixed hyperparameter-search pipeline. The correction
path must be autonomous.

Example:

``` text
Baseline macro F1: 0.58

Agent:
"Missing-hole recall is extremely poor and the class is underrepresented."

Retrieved MongoDB memory:
"Similar imbalance previously improved with weighted sampling."

Agent:
"Use weighted sampling."

New F1: 0.69

Critic:
"Minority recall improved significantly, but small open circuits remain difficult."

Agent:
"Increase crop resolution and augmentation."

New F1: 0.78
```

Every decision and outcome becomes experience available to future runs.

## Killer Evaluation

### Run A --- Cold Start

``` text
Experiment 1 → 61%
Experiment 2 → 68%
Experiment 3 → 74%
Experiment 4 → 81%
Experiment 5 → 85%
```

### Run B --- Experienced Agent

Give it a related PCB classification problem. MongoDB retrieves relevant
lessons from Run A.

``` text
Experiment 1 → 74%
Experiment 2 → 85%
```

The key result:

> **Cold-start agent required 5 experiments. Experienced agent required
> 2.**

This proves persistent memory changed the agent's behavior rather than
simply providing additional prompt context.

## Priority

Build backend only first.

Do not spend time on frontend, authentication, dashboards, or
unnecessary infrastructure until the autonomous loop works end-to-end.

### First milestone

> **A LangGraph agent persisted by MongoDB that retrieves previous ML
> experience, lets Fireworks choose the next real experiment, executes
> it, evaluates the outcome, and writes the resulting lesson back into
> MongoDB.**
