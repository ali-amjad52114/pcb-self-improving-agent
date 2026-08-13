"""Agent memory API — Person 1 contract for LangGraph (Person 3) and Fireworks (Person 2).

Deep Atlas usage:
- JSON Schema validation
- Voyage embeddings + $vectorSearch
- Atlas Search lexical retrieval
- $rankFusion hybrid search (MongoDB 8+)
- transactions when committing experiment + lesson together
- computed run_summaries
- aggregations for run history / best interventions
- change streams for live lesson writes
- LangGraph MongoDB checkpointer helper
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterator
from uuid import uuid4

from pymongo import ASCENDING, DESCENDING, ReturnDocument
from pymongo.database import Database
from pymongo.errors import CollectionInvalid, OperationFailure
from pymongo.operations import SearchIndexModel

from pcb_memory.client import get_db
from pcb_memory.embed import embed_document, embed_query
from pcb_memory.schema import (
    EXPERIMENT_VALIDATOR,
    LESSON_VALIDATOR,
    LEXICAL_INDEX,
    VECTOR_INDEX,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:10]}"


class AgentMemory:
    def __init__(self, db: Database | None = None):
        self.db = db or get_db()
        self.experiments = self.db["experiments"]
        self.lessons = self.db["lessons"]
        self.run_summaries = self.db["run_summaries"]
        self.checkpoints = self.db["langgraph_checkpoints"]

    def bootstrap(self) -> dict[str, Any]:
        """Create collections, validators, indexes, and Atlas Search indexes."""
        created = []
        for name, validator in (
            ("experiments", EXPERIMENT_VALIDATOR),
            ("lessons", LESSON_VALIDATOR),
        ):
            try:
                self.db.create_collection(name, validator=validator, validationLevel="moderate")
                created.append(name)
            except CollectionInvalid:
                self.db.command(
                    "collMod",
                    name,
                    validator=validator,
                    validationLevel="moderate",
                )
        self.db["run_summaries"]
        self.experiments.create_index([("experiment_id", ASCENDING)], unique=True)
        self.experiments.create_index([("run_id", ASCENDING), ("iteration", ASCENDING)])
        self.experiments.create_index([("status", ASCENDING), ("created_at", DESCENDING)])
        self.lessons.create_index([("lesson_id", ASCENDING)], unique=True)
        self.lessons.create_index([("run_id", ASCENDING), ("created_at", DESCENDING)])
        self.lessons.create_index([("defect_family", ASCENDING), ("outcome", ASCENDING)])
        self.run_summaries.create_index([("run_id", ASCENDING)], unique=True)

        search_indexes = []
        for spec in (VECTOR_INDEX, LEXICAL_INDEX):
            try:
                self.lessons.create_search_index(
                    SearchIndexModel(
                        definition=spec["definition"],
                        name=spec["name"],
                        type=spec["type"],
                    )
                )
                search_indexes.append(spec["name"])
            except OperationFailure as exc:
                search_indexes.append(f"{spec['name']} (exists or pending: {exc})")

        return {"collections": created or "already existed", "search_indexes": search_indexes}

    def store_experiment(self, data: dict) -> str:
        experiment_id = data.get("experiment_id") or _id("exp")
        doc = {
            **data,
            "experiment_id": experiment_id,
            "run_id": data.get("run_id") or _id("run"),
            "status": data.get("status") or "complete",
            "schema_version": 1,
            "created_at": data.get("created_at") or _now(),
            "updated_at": _now(),
        }
        self.experiments.replace_one({"experiment_id": experiment_id}, doc, upsert=True)
        self._refresh_run_summary(doc["run_id"])
        return experiment_id

    def store_lesson(self, data: dict) -> str:
        lesson_id = data.get("lesson_id") or _id("lesson")
        failure = data["failure_summary"]
        intervention = data["intervention"]
        result = data.get("result") or data.get("lesson") or ""
        embed_source = f"{failure}\n{intervention}\n{result}"
        doc = {
            **data,
            "lesson_id": lesson_id,
            "failure_summary": failure,
            "intervention": intervention,
            "result": result,
            "outcome": data.get("outcome") or "mixed",
            "defect_family": data.get("defect_family") or "unknown",
            "embedding": data.get("embedding") or embed_document(embed_source),
            "schema_version": 1,
            "created_at": data.get("created_at") or _now(),
        }
        self.lessons.replace_one({"lesson_id": lesson_id}, doc, upsert=True)
        return lesson_id

    def commit_experience(self, experiment: dict, lesson: dict) -> dict[str, str]:
        """Atomic write: experiment + lesson in one transaction."""
        with self.db.client.start_session() as session:
            with session.start_transaction():
                experiment_id = experiment.get("experiment_id") or _id("exp")
                lesson_id = lesson.get("lesson_id") or _id("lesson")
                experiment = {**experiment, "experiment_id": experiment_id}
                lesson = {
                    **lesson,
                    "lesson_id": lesson_id,
                    "experiment_id": experiment_id,
                    "run_id": lesson.get("run_id") or experiment.get("run_id"),
                }
                exp_doc = {
                    **experiment,
                    "status": experiment.get("status") or "complete",
                    "schema_version": 1,
                    "created_at": experiment.get("created_at") or _now(),
                    "updated_at": _now(),
                }
                failure = lesson["failure_summary"]
                intervention = lesson["intervention"]
                result = lesson.get("result") or lesson.get("lesson") or ""
                lesson_doc = {
                    **lesson,
                    "result": result,
                    "outcome": lesson.get("outcome") or "mixed",
                    "defect_family": lesson.get("defect_family") or "unknown",
                    "embedding": lesson.get("embedding")
                    or embed_document(f"{failure}\n{intervention}\n{result}"),
                    "schema_version": 1,
                    "created_at": lesson.get("created_at") or _now(),
                }
                self.experiments.replace_one(
                    {"experiment_id": experiment_id}, exp_doc, upsert=True, session=session
                )
                self.lessons.replace_one(
                    {"lesson_id": lesson_id}, lesson_doc, upsert=True, session=session
                )
        self._refresh_run_summary(experiment.get("run_id") or lesson.get("run_id"))
        return {"experiment_id": experiment_id, "lesson_id": lesson_id}

    def retrieve_similar_lessons(
        self,
        query: str,
        k: int = 5,
        *,
        defect_family: str | None = None,
        outcome: str | None = None,
        hybrid: bool = True,
    ) -> list[dict]:
        """Semantic + lexical hybrid retrieval. Falls back to vector-only if fusion fails."""
        vector = embed_query(query)
        vec_filter: dict[str, Any] = {}
        if defect_family:
            vec_filter["defect_family"] = defect_family
        if outcome:
            vec_filter["outcome"] = outcome

        vector_pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX["name"],
                    "path": "embedding",
                    "queryVector": vector,
                    "numCandidates": max(50, k * 15),
                    "limit": k,
                    **({"filter": vec_filter} if vec_filter else {}),
                }
            },
            {"$addFields": {"vs_score": {"$meta": "vectorSearchScore"}}},
        ]
        lexical_must = [{"text": {"query": query, "path": ["failure_summary", "intervention", "result"]}}]
        if defect_family:
            lexical_must.append({"equals": {"path": "defect_family", "value": defect_family}})
        if outcome:
            lexical_must.append({"equals": {"path": "outcome", "value": outcome}})
        lexical_pipeline = [
            {
                "$search": {
                    "index": LEXICAL_INDEX["name"],
                    "compound": {"must": lexical_must},
                }
            },
            {"$limit": k},
            {"$addFields": {"lex_score": {"$meta": "searchScore"}}},
        ]
        projection = {
            "$project": {
                "embedding": 0,
                "score": {"$meta": "vectorSearchScore"},
            }
        }

        if hybrid:
            try:
                fusion = [
                    {
                        "$rankFusion": {
                            "input": {
                                "pipelines": {
                                    "vector": vector_pipeline,
                                    "lexical": lexical_pipeline,
                                }
                            },
                            "combination": {"weights": {"vector": 0.7, "lexical": 0.3}},
                        }
                    },
                    {"$limit": k},
                    {"$project": {"embedding": 0}},
                ]
                return list(self.lessons.aggregate(fusion))
            except OperationFailure:
                pass

        pipeline = vector_pipeline + [projection, {"$limit": k}]
        return list(self.lessons.aggregate(pipeline))

    def get_run_history(self, run_id: str) -> list[dict]:
        return list(
            self.experiments.aggregate(
                [
                    {"$match": {"run_id": run_id}},
                    {"$sort": {"iteration": 1, "created_at": 1}},
                    {
                        "$lookup": {
                            "from": "lessons",
                            "localField": "experiment_id",
                            "foreignField": "experiment_id",
                            "as": "lessons",
                            "pipeline": [{"$project": {"embedding": 0}}],
                        }
                    },
                    {"$project": {"_id": 0}},
                ]
            )
        )

    def best_interventions(self, defect_family: str | None = None, limit: int = 5) -> list[dict]:
        """Computed-pattern query: which interventions actually moved F1."""
        match: dict[str, Any] = {"delta": {"$exists": True}}
        if defect_family:
            match["defect_family"] = defect_family
        return list(
            self.lessons.aggregate(
                [
                    {"$match": match},
                    {
                        "$group": {
                            "_id": "$intervention",
                            "avg_delta": {"$avg": "$delta"},
                            "wins": {"$sum": {"$cond": [{"$eq": ["$outcome", "helped"]}, 1, 0]}},
                            "n": {"$sum": 1},
                        }
                    },
                    {"$sort": {"avg_delta": -1}},
                    {"$limit": limit},
                ]
            )
        )

    def watch_lessons(self) -> Iterator[dict]:
        """Change stream — Person 3 can react as soon as a lesson is written."""
        with self.lessons.watch([{"$match": {"operationType": {"$in": ["insert", "replace", "update"]}}}]) as stream:
            for change in stream:
                yield change

    def checkpointer(self):
        """LangGraph MongoDB checkpointer (Person 3 wires this into the graph)."""
        from langgraph.checkpoint.mongodb import MongoDBSaver

        return MongoDBSaver(self.db.client, db_name=self.db.name, collection_name="langgraph_checkpoints")

    def _refresh_run_summary(self, run_id: str | None) -> None:
        if not run_id:
            return
        summary = next(
            self.experiments.aggregate(
                [
                    {"$match": {"run_id": run_id}},
                    {
                        "$group": {
                            "_id": "$run_id",
                            "n": {"$sum": 1},
                            "last_status": {"$last": "$status"},
                            "best_f1": {"$max": "$metrics.macro_f1"},
                            "updated_at": {"$max": "$updated_at"},
                        }
                    },
                ]
            ),
            None,
        )
        if not summary:
            return
        self.run_summaries.find_one_and_update(
            {"run_id": run_id},
            {
                "$set": {
                    "run_id": run_id,
                    "experiment_count": summary["n"],
                    "last_status": summary["last_status"],
                    "best_f1": summary.get("best_f1"),
                    "updated_at": _now(),
                }
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )


def store_experiment(data: dict) -> str:
    return AgentMemory().store_experiment(data)


def store_lesson(data: dict) -> str:
    return AgentMemory().store_lesson(data)


def retrieve_similar_lessons(query: str, k: int = 5) -> list:
    return AgentMemory().retrieve_similar_lessons(query, k=k)


def get_run_history(run_id: str) -> list:
    return AgentMemory().get_run_history(run_id)
