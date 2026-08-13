from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from scripts import bootstrap_memory


class BootstrapMemoryTest(unittest.TestCase):
    def test_seed_data_uses_atomic_linked_experience_writes(self) -> None:
        memory = MagicMock(name="memory")
        memory.bootstrap.return_value = {"collections": [], "search_indexes": []}
        memory.retrieve_similar_lessons.return_value = []

        with patch.object(bootstrap_memory, "AgentMemory", return_value=memory), patch("builtins.print"):
            bootstrap_memory.main()

        self.assertEqual(memory.commit_experience.call_count, len(bootstrap_memory.SEED_LESSONS))
        memory.store_experiment.assert_not_called()
        memory.store_lesson.assert_not_called()
        for call, seed in zip(memory.commit_experience.call_args_list, bootstrap_memory.SEED_LESSONS):
            experiment = call.kwargs["experiment"]
            lesson = call.kwargs["lesson"]
            self.assertEqual(experiment["run_id"], seed["run_id"])
            self.assertEqual(lesson["run_id"], seed["run_id"])
            self.assertEqual(lesson["failure_summary"], seed["failure_summary"])


if __name__ == "__main__":
    unittest.main()
