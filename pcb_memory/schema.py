"""JSON Schema validators — keep agent memory documents consistent."""

NUMBER_TYPES = ["double", "int", "long", "decimal"]
F1_SCORE = {"bsonType": NUMBER_TYPES, "minimum": 0, "maximum": 1}


EXPERIMENT_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["experiment_id", "run_id", "status", "created_at"],
        "properties": {
            "experiment_id": {"bsonType": "string", "minLength": 1},
            "run_id": {"bsonType": "string", "minLength": 1},
            "iteration": {"bsonType": ["int", "long"], "minimum": 0},
            "dataset_stats": {"bsonType": "object"},
            "model_config": {"bsonType": "object"},
            "metrics": {"bsonType": "object"},
            "confusion_matrix": {"bsonType": "object"},
            "failure_signature": {"bsonType": "object"},
            "intervention": {"bsonType": "object"},
            "outcome_delta": {"bsonType": "object"},
            "status": {"enum": ["proposed", "running", "complete", "failed"]},
            "schema_version": {"bsonType": ["int", "long"], "minimum": 1},
            "created_at": {"bsonType": "date"},
            "updated_at": {"bsonType": "date"},
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
            "lesson_id": {"bsonType": "string", "minLength": 1},
            "run_id": {"bsonType": "string", "minLength": 1},
            "experiment_id": {"bsonType": "string", "minLength": 1},
            "failure_summary": {"bsonType": "string", "minLength": 1},
            "intervention": {"bsonType": "string", "minLength": 1},
            "result": {"bsonType": "string"},
            "outcome": {"enum": ["helped", "failed", "mixed"]},
            "defect_family": {"bsonType": "string", "minLength": 1},
            "before_f1": F1_SCORE,
            "after_f1": F1_SCORE,
            "delta": {"bsonType": NUMBER_TYPES, "minimum": -1, "maximum": 1},
            "confidence": F1_SCORE,
            "embedding": {
                "bsonType": "array",
                "minItems": 1024,
                "maxItems": 1024,
                "items": {"bsonType": NUMBER_TYPES},
            },
            "schema_version": {"bsonType": ["int", "long"], "minimum": 1},
            "created_at": {"bsonType": "date"},
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
