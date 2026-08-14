"""JSON Schemas for every structured response we ask Fireworks for.

These are plain dicts rather than Pydantic models on purpose. Fireworks'
`response_format={"type": "json_schema", ...}` takes raw JSON Schema, the
project brief specifies plain-JSON interfaces between the three owners, and
these same dicts go straight into MongoDB validators if Person 1 wants them.

Every schema sets `additionalProperties: false` so the constrained decoder
cannot invent fields, and lists `required` explicitly so partial objects fail
loudly instead of silently defaulting.
"""

from __future__ import annotations

from typing import Any

#: The experiment actions the scientist may choose. Kept in one place because
#: `tools.py` derives the tool allowlist from it and the schemas below
#: reference it as an enum.
ACTIONS: tuple[str, ...] = (
    "change_model",
    "change_learning_rate",
    "change_batch_size",
    "change_image_size",
    "change_augmentation",
    "change_sampler",
    "change_class_weights",
    "change_confidence_threshold",
    "retrain",
)

CONFIDENCE = {
    "type": "number",
    "minimum": 0.0,
    "maximum": 1.0,
    "description": "Calibrated confidence in this judgement, 0 to 1.",
}


DIAGNOSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["diagnosis", "failure_signature", "hypothesis", "confidence"],
    "properties": {
        "diagnosis": {
            "type": "string",
            "description": "Why the classifier is underperforming, in one or two sentences.",
        },
        "failure_signature": {
            "type": "object",
            "additionalProperties": False,
            "required": ["summary", "affected_classes", "failure_mode"],
            "description": "Compact, embeddable description of the failure. This is "
                           "the text used as the Atlas Vector Search query key.",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "One sentence, written so a future run with a "
                                   "similar problem retrieves it by similarity.",
                },
                "affected_classes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Defect class names that are failing worst.",
                },
                "failure_mode": {
                    "type": "string",
                    "enum": [
                        "class_imbalance",
                        "low_recall_minority",
                        "low_precision_overprediction",
                        "small_object_detection",
                        "confusable_class_pair",
                        "underfitting",
                        "overfitting",
                        "threshold_miscalibration",
                        "data_quality",
                        "unclear",
                    ],
                },
                "confusion_pairs": {
                    "type": "array",
                    "description": "Class pairs the model conflates, worst first.",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["true_class", "predicted_class"],
                        "properties": {
                            "true_class": {"type": "string"},
                            "predicted_class": {"type": "string"},
                            "count": {"type": "integer", "minimum": 0},
                        },
                    },
                },
            },
        },
        "hypothesis": {
            "type": "string",
            "description": "A falsifiable statement about what change would help and why.",
        },
        "confidence": CONFIDENCE,
        "evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific numbers from the metrics or confusion matrix "
                           "that support the diagnosis.",
        },
        "memory_used": {
            "type": "array",
            "items": {"type": "string"},
            "description": "lesson_id values from retrieved memory that informed this "
                           "diagnosis. Empty on a cold start.",
        },
        "ruled_out": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Interventions deliberately not chosen, and why. Populated "
                           "from retrieved lessons that previously failed.",
        },
    },
}


PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["next_action", "parameters", "expected_effect", "confidence"],
    "properties": {
        "next_action": {"type": "string", "enum": list(ACTIONS)},
        "parameters": {
            "type": "object",
            "description": "Arguments for the chosen action. Must satisfy that "
                           "action's tool schema in tools.py.",
        },
        "expected_effect": {
            "type": "string",
            "description": "The metric that should move and roughly by how much.",
        },
        "expected_delta_macro_f1": {
            "type": "number",
            "minimum": -1.0,
            "maximum": 1.0,
            "description": "Predicted change in macro F1. Recorded so the critic can "
                           "score the scientist's calibration over time.",
        },
        "confidence": CONFIDENCE,
        "rationale": {"type": "string"},
        "second_opinion_requested": {
            "type": "boolean",
            "description": "True when the scientist wants the OpenRouter judge to "
                           "check this before it is executed.",
        },
    },
}


CRITIQUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["hypothesis_correct", "verdict", "explanation", "lesson", "confidence"],
    "properties": {
        "hypothesis_correct": {"type": "boolean"},
        "verdict": {
            "type": "string",
            "enum": ["improved", "regressed", "no_change", "mixed"],
        },
        "explanation": {
            "type": "string",
            "description": "Why the intervention helped or failed, grounded in the "
                           "before/after numbers.",
        },
        "lesson": {
            "type": "object",
            "additionalProperties": False,
            "required": ["failure_summary", "intervention", "result", "lesson", "reusable"],
            "description": "The document written to the MongoDB `lessons` collection.",
            "properties": {
                "failure_summary": {
                    "type": "string",
                    "description": "The problem this lesson is about, phrased so a "
                                   "future failure_signature.summary matches it.",
                },
                "intervention": {"type": "string"},
                "result": {
                    "type": "string",
                    "description": "Human-readable outcome, e.g. '+11% macro F1'.",
                },
                "lesson": {
                    "type": "string",
                    "description": "Transferable advice. Must not mention run ids or "
                                   "experiment numbers.",
                },
                "reusable": {
                    "type": "boolean",
                    "description": "False for one-off flukes that should not steer "
                                   "future runs.",
                },
                "avoid": {
                    "type": "boolean",
                    "description": "True when this records something that did NOT work, "
                                   "so future runs skip it.",
                },
            },
        },
        "confidence": CONFIDENCE,
        "next_focus": {
            "type": "string",
            "description": "What the next experiment should target, given this outcome.",
        },
    },
}


#: The scientist's failure signature is what gets embedded and stored. Keeping
#: the embeddable text generation as its own schema lets us use the cheap FAST
#: model for it instead of burning scientist tokens.
FAILURE_SIGNATURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "keywords"],
    "properties": {
        "summary": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
    },
}


VISION_REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["observations", "likely_cause", "suggested_focus"],
    "properties": {
        "observations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "What is visually true of the misclassified crops.",
        },
        "defect_scale": {
            "type": "string",
            "enum": ["sub_pixel", "small", "medium", "large", "mixed", "unclear"],
        },
        "image_quality_issues": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["blur", "low_contrast", "glare", "occlusion",
                         "crop_too_wide", "crop_too_tight", "none"],
            },
        },
        "likely_cause": {"type": "string"},
        "suggested_focus": {
            "type": "string",
            "description": "Which knob this evidence points at, in plain language. "
                           "The scientist decides the actual action.",
        },
        "confidence": CONFIDENCE,
    },
}


#: Independent judge output. Person 3 runs this through OpenRouter, but the
#: schema lives here so both providers are asked for the identical shape and
#: their answers are directly comparable.
SECOND_OPINION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["agrees", "confidence", "reasoning"],
    "properties": {
        "agrees": {"type": "boolean"},
        "confidence": CONFIDENCE,
        "reasoning": {"type": "string"},
        "alternative_diagnosis": {"type": "string"},
        "alternative_action": {"type": "string", "enum": list(ACTIONS)},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
    },
}


ALL_SCHEMAS: dict[str, dict[str, Any]] = {
    "Diagnosis": DIAGNOSIS_SCHEMA,
    "ExperimentProposal": PROPOSAL_SCHEMA,
    "Critique": CRITIQUE_SCHEMA,
    "FailureSignature": FAILURE_SIGNATURE_SCHEMA,
    "VisionReport": VISION_REPORT_SCHEMA,
    "SecondOpinion": SECOND_OPINION_SCHEMA,
}
