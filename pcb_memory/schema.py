"""JSON Schema validators — keep agent memory documents consistent."""

EXPERIMENT_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["experiment_id", "run_id", "status", "created_at"],
        "properties": {
            "experiment_id": {"bsonType": "string"},
            "run_id": {"bsonType": "string"},
            "iteration": {"bsonType": ["int", "long"]},
            "dataset_stats": {"bsonType": "object"},
            "model_config": {"bsonType": "object"},
            "metrics": {"bsonType": "object"},
            "confusion_matrix": {"bsonType": "object"},
            "failure_signature": {"bsonType": "object"},
            "intervention": {"bsonType": "object"},
            "outcome_delta": {"bsonType": "object"},
            "status": {"enum": ["proposed", "running", "complete", "failed"]},
            "schema_version": {"bsonType": ["int", "long"]},
        },
    }
}

LESSON_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": [
            "lesson_id",
            "failure_summary",
            "intervention",
            "result",
            "embedding",
            "created_at",
        ],
        "properties": {
            "lesson_id": {"bsonType": "string"},
            "run_id": {"bsonType": "string"},
            "experiment_id": {"bsonType": "string"},
            "failure_summary": {"bsonType": "string"},
            "intervention": {"bsonType": "string"},
            "result": {"bsonType": "string"},
            "outcome": {"enum": ["helped", "failed", "mixed"]},
            "defect_family": {"bsonType": "string"},
            "before_f1": {"bsonType": ["double", "int", "long", "decimal"]},
            "after_f1": {"bsonType": ["double", "int", "long", "decimal"]},
            "delta": {"bsonType": ["double", "int", "long", "decimal"]},
            "confidence": {"bsonType": ["double", "int", "long", "decimal"]},
            "embedding": {"bsonType": "array"},
            "schema_version": {"bsonType": ["int", "long"]},
        },
    }
}

VECTOR_INDEX = {
    "name": "lessons_vector_index",
    "type": "vectorSearch",
    "definition": {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 1024,
                "similarity": "cosine",
            },
            {"type": "filter", "path": "outcome"},
            {"type": "filter", "path": "defect_family"},
            {"type": "filter", "path": "run_id"},
        ]
    },
}

LEXICAL_INDEX = {
    "name": "lessons_lexical_index",
    "type": "search",
    "definition": {
        "mappings": {
            "dynamic": False,
            "fields": {
                "failure_summary": {
                    "type": "string",
                    "analyzer": "lucene.english",
                },
                "intervention": {"type": "string", "analyzer": "lucene.english"},
                "result": {"type": "string", "analyzer": "lucene.english"},
                "defect_family": {"type": "token"},
                "outcome": {"type": "token"},
            },
        }
    },
}

EMBED_MODEL = "voyage-3-large"
EMBED_DIMS = 1024
