# Person 1 MongoDB MVP handoff

## What the demo proves

Person 1 owns the persistent experience boundary. One command demonstrates that MongoDB Atlas is part of the agent's decision loop, rather than a passive event log:

```powershell
python -m scripts.demo_atlas_memory
```

The command produces a single JSON artifact with evidence of:

1. Atlas connectivity and the live collection/search-index state.
2. Atomic, linked writes of an experiment and its distilled lesson.
3. Voyage embeddings stored in Atlas and hybrid Vector Search + lexical retrieval for a new failure.
4. `$lookup` reconstruction of the complete run history.
5. An aggregation ranking interventions by observed F1 delta.
6. An Atlas-computed learning curve and target-reaching iteration.
7. A materialized run summary with experiment count and best F1.

The two demo experiences use fixed IDs. Rerunning the command reuses them instead of adding duplicate records. Use a custom current failure like this:

```powershell
python -m scripts.demo_atlas_memory --query "tiny open-circuit defects disappear in coarse image crops"
```

Atlas Search indexes can take a short time to become queryable after first creation. The command waits up to 60 seconds for both indexes. If Atlas is still building them after that, rerun with `--skip-bootstrap` once the Atlas UI shows them as active.

## The integration contract

Person 2 does not write MongoDB plumbing. The scientist receives the result of:

```python
lessons = memory.retrieve_similar_lessons(failure_summary, k=5)
```

Those lessons become evidence in its next-experiment prompt. After evaluation and critique, it returns ordinary dictionaries for the memory layer to persist.

Person 3 does not implement storage. Graph nodes call:

```python
memory.commit_experience(experiment, lesson)
history = memory.get_run_history(run_id)
comparison = memory.compare_runs([cold_run_id, experienced_run_id], target_f1=0.84)
```

Person 3 may opt into `memory.checkpointer()` when wiring the graph, but Person 1 does not construct or control the graph.

## Ownership boundary

| Person 1 owns | Person 1 does not own |
|---|---|
| Atlas collections, validators, indexes, and retrieval | Model training or evaluation |
| Experiment and lesson persistence | Fireworks prompts or experiment selection |
| Voyage embedding and hybrid search | LangGraph state, edges, retries, or stopping |
| Run-history and intervention aggregations | UI, deployment, or production hardening |
| MongoDB checkpointer factory | Wiring the checkpointer into the graph |

## Judge-facing story

> The classifier learns weights during one run. The agent learns how to improve classifiers across runs. Every attempted intervention and measured outcome becomes structured Atlas memory. When the next failure appears, MongoDB retrieves semantically and lexically related successes and failures, so the agent can repeat what worked and avoid what failed. The demo then uses Atlas aggregations to show that this experience is measurable, linked, and reusable.

For a concise live demo, show `retrieval.hits`, `linked_run_history`,
`computed_best_interventions`, and `atlas_learning_evidence` from the command
output, then point to the corresponding documents and Vector Search indexes in
Atlas. Once Persons 2 and 3 produce the real cold and experienced runs, pass
both run IDs to `compare_runs()` to calculate the headline result in MongoDB.
