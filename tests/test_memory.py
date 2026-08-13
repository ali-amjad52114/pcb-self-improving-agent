from __future__ import annotations

import unittest
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

from pymongo.errors import OperationFailure

from pcb_memory.memory import AgentMemory
from pcb_memory.schema import LEXICAL_INDEX, VECTOR_INDEX


class FakeDatabase:
    """Small database double that preserves collection identity."""

    def __init__(self) -> None:
        self.name = "test_memory"
        self.client = MagicMock(name="mongo_client")
        self.collections: dict[str, MagicMock] = {}

    def __getitem__(self, name: str) -> MagicMock:
        return self.collections.setdefault(name, MagicMock(name=name))


class AgentMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.db = FakeDatabase()
        self.memory = AgentMemory(self.db)

    @patch("pcb_memory.memory.embed_document", return_value=[0.1, 0.2])
    def test_commit_experience_atomically_links_lesson_to_experiment(self, embed: MagicMock) -> None:
        session = MagicMock(name="session")
        session.__enter__.return_value = session
        session.start_transaction.return_value = nullcontext()
        self.db.client.start_session.return_value = session

        with patch.object(self.memory, "_refresh_run_summary") as refresh:
            ids = self.memory.commit_experience(
                {
                    "experiment_id": "exp_1",
                    "run_id": "run_1",
                    "iteration": 1,
                    "metrics": {"macro_f1": 0.78},
                },
                {
                    "lesson_id": "lesson_1",
                    "failure_summary": "open circuits have poor recall",
                    "intervention": "weighted sampler",
                    "result": "macro F1 improved",
                },
            )

        self.assertEqual(ids, {"experiment_id": "exp_1", "lesson_id": "lesson_1"})
        exp_doc = self.memory.experiments.replace_one.call_args.args[1]
        lesson_doc = self.memory.lessons.replace_one.call_args.args[1]
        self.assertEqual(lesson_doc["experiment_id"], exp_doc["experiment_id"])
        self.assertEqual(lesson_doc["run_id"], exp_doc["run_id"])
        self.assertEqual(self.memory.experiments.replace_one.call_args.kwargs["session"], session)
        self.assertEqual(self.memory.lessons.replace_one.call_args.kwargs["session"], session)
        embed.assert_called_once_with(
            "open circuits have poor recall\nweighted sampler\nmacro F1 improved"
        )
        refresh.assert_called_once_with("run_1")

    @patch("pcb_memory.memory.embed_query", return_value=[0.25, 0.75])
    def test_hybrid_retrieval_builds_rank_fusion_with_filters(self, embed: MagicMock) -> None:
        self.memory.lessons.aggregate.return_value = iter([{"lesson_id": "lesson_1"}])

        hits = self.memory.retrieve_similar_lessons(
            "poor open circuit recall",
            k=3,
            defect_family="open_circuit",
            outcome="helped",
        )

        self.assertEqual(hits, [{"lesson_id": "lesson_1"}])
        embed.assert_called_once_with("poor open circuit recall")
        pipeline = self.memory.lessons.aggregate.call_args.args[0]
        fusion = pipeline[0]["$rankFusion"]
        self.assertEqual(fusion["combination"]["weights"], {"vector": 0.7, "lexical": 0.3})
        for input_pipeline in fusion["input"]["pipelines"].values():
            self.assertFalse(
                any("$addFields" in stage or "$project" in stage for stage in input_pipeline),
                "$rankFusion input pipelines must return unmodified source documents",
            )
        vector_search = fusion["input"]["pipelines"]["vector"][0]["$vectorSearch"]
        self.assertEqual(vector_search["index"], VECTOR_INDEX["name"])
        self.assertEqual(vector_search["queryVector"], [0.25, 0.75])
        self.assertGreaterEqual(vector_search["numCandidates"], 3)
        self.assertGreaterEqual(vector_search["numCandidates"], vector_search["limit"])
        self.assertEqual(vector_search["filter"], {"defect_family": "open_circuit", "outcome": "helped"})
        search = fusion["input"]["pipelines"]["lexical"][0]["$search"]
        self.assertEqual(search["index"], LEXICAL_INDEX["name"])
        self.assertIn(
            {"equals": {"path": "defect_family", "value": "open_circuit"}},
            search["compound"]["filter"],
        )
        self.assertIn(
            {"equals": {"path": "outcome", "value": "helped"}},
            search["compound"]["filter"],
        )

    @patch("pcb_memory.memory.embed_query", return_value=[0.5])
    def test_hybrid_retrieval_falls_back_to_vector_search(self, _embed: MagicMock) -> None:
        self.memory.lessons.aggregate.side_effect = [
            OperationFailure("rank fusion unavailable"),
            iter([{"lesson_id": "fallback"}]),
        ]

        hits = self.memory.retrieve_similar_lessons("failure", k=2)

        self.assertEqual(hits, [{"lesson_id": "fallback"}])
        self.assertEqual(self.memory.lessons.aggregate.call_count, 2)
        fallback = self.memory.lessons.aggregate.call_args.args[0]
        self.assertIn("$vectorSearch", fallback[0])
        self.assertEqual(fallback[0]["$vectorSearch"]["limit"], 2)
        self.assertEqual(fallback[-1], {"$limit": 2})

    def test_run_history_joins_lessons_by_experiment_id_without_embeddings(self) -> None:
        self.memory.experiments.aggregate.return_value = iter([])

        self.memory.get_run_history("run_1")

        pipeline = self.memory.experiments.aggregate.call_args.args[0]
        self.assertEqual(pipeline[0], {"$match": {"run_id": "run_1"}})
        lookup = next(stage["$lookup"] for stage in pipeline if "$lookup" in stage)
        self.assertEqual(lookup["localField"], "experiment_id")
        self.assertEqual(lookup["foreignField"], "experiment_id")
        self.assertEqual(lookup["pipeline"], [{"$project": {"embedding": 0}}])

    def test_run_summary_sorts_before_using_last_status(self) -> None:
        self.memory.experiments.aggregate.return_value = iter(
            [{"_id": "run_1", "n": 2, "last_status": "complete", "best_f1": 0.82}]
        )

        self.memory._refresh_run_summary("run_1")

        pipeline = self.memory.experiments.aggregate.call_args.args[0]
        group_index = next(i for i, stage in enumerate(pipeline) if "$group" in stage)
        sort_stages = [stage["$sort"] for stage in pipeline[:group_index] if "$sort" in stage]
        self.assertTrue(
            any(sort.get("iteration") == 1 or sort.get("updated_at") == 1 for sort in sort_stages),
            "$last is nondeterministic unless experiments are sorted before $group",
        )

    def test_checkpointer_uses_named_checkpoint_and_write_collections(self) -> None:
        saver = MagicMock(name="MongoDBSaver")

        with patch("langgraph.checkpoint.mongodb.MongoDBSaver", saver):
            self.memory.checkpointer()

        kwargs = saver.call_args.kwargs
        self.assertEqual(kwargs["db_name"], "test_memory")
        self.assertEqual(kwargs["checkpoint_collection_name"], "langgraph_checkpoints")
        self.assertEqual(kwargs["writes_collection_name"], "langgraph_checkpoint_writes")
        self.assertNotIn("collection_name", kwargs)

    def test_compare_runs_builds_cold_start_evidence_in_atlas(self) -> None:
        self.memory.experiments.aggregate.return_value = iter([])

        self.memory.compare_runs(["cold", "experienced"], target_f1=0.84)

        pipeline = self.memory.experiments.aggregate.call_args.args[0]
        self.assertEqual(pipeline[0]["$match"]["run_id"], {"$in": ["cold", "experienced"]})
        group = next(stage["$group"] for stage in pipeline if "$group" in stage)
        self.assertIn("learning_curve", group)
        set_stage = next(stage["$set"] for stage in pipeline if "$set" in stage)
        threshold = set_stage["first_target_iteration"]["$min"]["$map"]["input"]["$filter"]["cond"]
        self.assertEqual(threshold, {"$gte": ["$$point.macro_f1", 0.84]})


if __name__ == "__main__":
    unittest.main()
