from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from pcb_agent.integrations.external_memory import ExternalMemoryAdapter
from pcb_agent.nodes.persistence import make_store_experience


class Person3MongoDBIntegrationTest(unittest.TestCase):
    def test_external_adapter_prefers_person1_atomic_commit(self) -> None:
        module = MagicMock()
        module.commit_experience.return_value = {
            "experiment_id": "exp_1",
            "lesson_id": "lesson_1",
        }
        adapter = ExternalMemoryAdapter(module)

        result = adapter.commit_experience({"run_id": "run_1"}, {"result": "helped"})

        self.assertEqual(result["experiment_id"], "exp_1")
        module.commit_experience.assert_called_once()
        module.store_experiment.assert_not_called()
        module.store_lesson.assert_not_called()

    def test_persistence_maps_agent_outcome_into_searchable_atlas_lesson(self) -> None:
        memory = MagicMock()
        deps = SimpleNamespace(memory=memory)
        node = make_store_experience(deps)
        state = {
            "run_id": "run_1",
            "iteration": 0,
            "target_metric": "macro_f1",
            "previous_metrics": {"macro_f1": 0.55},
            "current_metrics": {"macro_f1": 0.74},
            "last_metric_delta": 0.19,
            "last_experiment_helped": True,
            "proposed_experiment": {"next_action": "change_sampler"},
            "critique": {
                "lesson": {
                    "failure_summary": "minority recall is poor",
                    "intervention": "weighted sampling",
                    "result": "recall improved",
                    "confidence": 0.9,
                }
            },
            "experiment_history": [],
            "retrieved_lessons": [],
            "warnings": [],
            "experiment_budget": 4,
            "target_value": 0.84,
        }

        node(state)

        experiment, lesson = memory.commit_experience.call_args.args
        self.assertEqual(experiment["experiment_id"], "run_1_exp_001")
        self.assertEqual(lesson["lesson_id"], "run_1_exp_001_lesson")
        self.assertEqual(lesson["outcome"], "helped")
        self.assertEqual(lesson["before_f1"], 0.55)
        self.assertEqual(lesson["after_f1"], 0.74)
        self.assertAlmostEqual(lesson["delta"], 0.19)


if __name__ == "__main__":
    unittest.main()
