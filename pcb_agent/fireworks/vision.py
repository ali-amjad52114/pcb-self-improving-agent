"""Fireworks vision models inspecting the PCB crops the classifier got wrong.

This is the one place the agent gets information that is not already in the
metrics. A confusion matrix says "missing_hole recall is 0.41". It cannot say
"the crops are 12 pixels wide at this input resolution" or "half of them are
out of focus". Those two observations point at completely different actions
(`change_image_size` vs a data-quality flag), and a text-only scientist has no
way to tell them apart.

So the flow is: evaluate -> collect the worst misclassified crops -> VLM report
-> feed the report into `diagnose` as extra evidence.

Fireworks image limits (from the docs): 30 images per request, base64 payload
under 10MB total, individual URLs under 5MB and fetched within 1.5s.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any, Mapping, Sequence

from .client import FireworksClient
from .errors import FireworksBadRequestError
from .models import Role
from .schemas import VISION_REPORT_SCHEMA
from .structured import complete_json

log = logging.getLogger("pcb_agent.fireworks.vision")

MAX_IMAGES_PER_REQUEST = 30
MAX_BASE64_PAYLOAD_BYTES = 10 * 1024 * 1024

VISION_SYSTEM = """\
You are inspecting crops from a printed circuit board defect dataset that the \
current classifier labelled incorrectly.

Report only what is visually verifiable. Judge defect scale relative to the \
crop, note image quality problems, and say which of these the evidence points \
at: input resolution, crop tightness, augmentation, or data quality.

Do not propose hyperparameters. Another component decides the action; you \
supply the visual evidence it cannot otherwise get.
"""

SUPPORTED_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".ppm",
}


def encode_image(path: str | Path) -> str:
    """Read a local image into a `data:` URI for the chat API."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FireworksBadRequestError(f"image not found: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise FireworksBadRequestError(
            f"unsupported image type {file_path.suffix!r}; Fireworks accepts "
            f"{sorted(SUPPORTED_SUFFIXES)}"
        )
    mime = mimetypes.guess_type(file_path.name)[0] or "image/png"
    payload = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def image_part(source: str | Path) -> dict[str, Any]:
    """One `image_url` content part. Accepts an http(s) URL, a data URI, or a path."""
    text = str(source)
    if text.startswith(("http://", "https://", "data:")):
        return {"type": "image_url", "image_url": {"url": text}}
    return {"type": "image_url", "image_url": {"url": encode_image(text)}}


def build_image_message(
    prompt: str, images: Sequence[str | Path], *, labels: Sequence[str] | None = None
) -> dict[str, Any]:
    """Build a multimodal user turn, enforcing the documented limits.

    `labels` interleaves a caption before each image, for example the true and
    predicted class, so the model can reason about *which* crop shows what.
    """
    if not images:
        raise FireworksBadRequestError("build_image_message requires at least one image")
    if len(images) > MAX_IMAGES_PER_REQUEST:
        raise FireworksBadRequestError(
            f"Fireworks accepts at most {MAX_IMAGES_PER_REQUEST} images per request, "
            f"got {len(images)}. Sample the worst crops instead."
        )

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    total_bytes = 0

    for index, source in enumerate(images):
        if labels is not None and index < len(labels):
            content.append({"type": "text", "text": f"Image {index + 1}: {labels[index]}"})
        part = image_part(source)
        url = part["image_url"]["url"]
        if url.startswith("data:"):
            total_bytes += len(url)
        content.append(part)

    if total_bytes > MAX_BASE64_PAYLOAD_BYTES:
        raise FireworksBadRequestError(
            f"base64 image payload is {total_bytes / 1e6:.1f}MB, over the "
            f"{MAX_BASE64_PAYLOAD_BYTES / 1e6:.0f}MB limit. Send fewer or smaller crops."
        )

    return {"role": "user", "content": content}


def inspect_misclassified(
    client: FireworksClient,
    images: Sequence[str | Path],
    *,
    labels: Sequence[str] | None = None,
    metrics_context: Mapping[str, Any] | None = None,
    role: Role = Role.VISION,
    max_images: int = 12,
) -> dict[str, Any]:
    """Produce a `VISION_REPORT_SCHEMA` report on misclassified crops.

    Defaults to 12 images rather than the API's 30: more crops mostly add
    latency, and 12 is enough to tell "systematically tiny" from "systematically
    blurry".
    """
    if not images:
        return {
            "observations": [],
            "likely_cause": "no crops supplied",
            "suggested_focus": "none",
            "confidence": 0.0,
            "skipped": True,
        }

    selected = list(images)[:max_images]
    selected_labels = list(labels)[:max_images] if labels else None

    prompt_parts = [
        f"These {len(selected)} crops were misclassified by the current model.",
    ]
    if metrics_context:
        import json
        prompt_parts.append(
            "Current metrics for context:\n"
            + json.dumps(metrics_context, indent=2, default=str)
        )
    prompt_parts.append(
        "Describe what the failures have in common visually, and say which of "
        "input resolution, crop tightness, augmentation, or data quality the "
        "evidence points at."
    )

    message = build_image_message(
        "\n\n".join(prompt_parts), selected, labels=selected_labels
    )

    report = complete_json(
        client,
        name="VisionReport",
        schema=VISION_REPORT_SCHEMA,
        messages=[{"role": "system", "content": VISION_SYSTEM}, message],
        role=role,
        purpose="vision_inspect",
    )
    report["images_inspected"] = len(selected)
    report["skipped"] = False
    return report


def describe_dataset_sample(
    client: FireworksClient,
    images: Sequence[str | Path],
    *,
    role: Role = Role.VISION_SMALL,
    max_images: int = 8,
) -> str:
    """A one-paragraph description of what the dataset looks like.

    Called once at run start and cached into `dataset_summary`, so the text-only
    scientist has some grounding in the imagery for every later iteration
    without paying for vision on each one.
    """
    if not images:
        return ""
    selected = list(images)[:max_images]
    message = build_image_message(
        "Describe this PCB defect dataset in one paragraph: board type, imaging "
        "conditions, typical defect size relative to the frame, and anything "
        "that would make classification hard.",
        selected,
    )
    payload = client.chat(
        [{"role": "system", "content": VISION_SYSTEM}, message],
        role=role,
        max_tokens=400,
        purpose="vision_dataset_summary",
    )
    return ((payload.get("choices") or [{}])[0].get("message") or {}).get("content", "")


def collect_worst_crops(
    predictions: Sequence[Mapping[str, Any]],
    *,
    limit: int = 12,
    image_field: str = "image_path",
) -> tuple[list[str], list[str]]:
    """Pick the most informative misclassified samples and their captions.

    "Most informative" means highest model confidence in the wrong answer:
    those are the systematic errors, not the borderline ones.
    """
    wrong = [
        p for p in predictions
        if p.get("true_label") != p.get("predicted_label") and p.get(image_field)
    ]
    wrong.sort(key=lambda p: float(p.get("confidence", 0.0)), reverse=True)
    chosen = wrong[:limit]

    paths = [str(p[image_field]) for p in chosen]
    captions = [
        f"true={p.get('true_label')} predicted={p.get('predicted_label')} "
        f"confidence={float(p.get('confidence', 0.0)):.2f}"
        for p in chosen
    ]
    return paths, captions
