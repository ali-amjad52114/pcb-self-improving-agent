"""Pure consensus merge for multi-model OpenRouter judge panels."""

from __future__ import annotations

from collections import Counter
from typing import Any

from pcb_agent.contracts import ALLOWED_ACTIONS, normalize_judge


def merge_panel_opinions(opinions: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge 1..N judge opinions into a consensus opinion.

    Reconsider rule:
    - recommendation=reconsider when >=2 members say reconsider, OR
    - a single disagree/reconsider vote has confidence >= 0.75 and no other
      proceed votes with higher confidence (single-model panels still work).
    """
    cleaned = [normalize_judge(o) for o in opinions if o]
    if not cleaned:
        from pcb_agent.contracts import JUDGE_FALLBACK

        return dict(JUDGE_FALLBACK)

    panel_rows: list[dict[str, Any]] = []
    for item in cleaned:
        panel_rows.append(
            {
                "model": item.get("_model") or item.get("model") or "unknown",
                "verdict": item.get("verdict"),
                "recommendation": item.get("recommendation"),
                "confidence": item.get("confidence"),
                "response_format_mode": item.get("_response_format_mode") or "",
            }
        )

    verdict_counts = Counter(str(o.get("verdict")) for o in cleaned)
    rec_counts = Counter(str(o.get("recommendation")) for o in cleaned)
    top_verdict, top_verdict_n = verdict_counts.most_common(1)[0]
    agreement_rate = top_verdict_n / max(1, len(cleaned))

    reconsider_votes = [
        o for o in cleaned if o.get("recommendation") == "reconsider"
    ]
    proceed_votes = [o for o in cleaned if o.get("recommendation") == "proceed"]

    mean_reconsider_conf = (
        sum(float(o.get("confidence") or 0.0) for o in reconsider_votes)
        / len(reconsider_votes)
        if reconsider_votes
        else 0.0
    )

    if len(reconsider_votes) >= 2 or (
        len(reconsider_votes) >= 1 and mean_reconsider_conf >= 0.75 and len(cleaned) == 1
    ):
        recommendation = "reconsider"
    elif len(reconsider_votes) >= 1 and mean_reconsider_conf >= 0.75 and len(proceed_votes) == 0:
        recommendation = "reconsider"
    elif len(reconsider_votes) >= 1 and mean_reconsider_conf >= 0.8:
        # Strong minority dissent can still force a single revision.
        recommendation = "reconsider"
    else:
        recommendation = "proceed"

    # Confidence: average of majority recommendation camp, else overall mean.
    camp = [
        o for o in cleaned if o.get("recommendation") == recommendation
    ] or cleaned
    confidence = sum(float(o.get("confidence") or 0.0) for o in camp) / len(camp)

    if recommendation == "reconsider" and top_verdict == "agree":
        verdict = "disagree"
    else:
        verdict = top_verdict

    dissenters = [
        o
        for o in cleaned
        if o.get("recommendation") != recommendation or o.get("verdict") != verdict
    ]
    dissent_summary = ""
    if dissenters:
        bits = [
            f"{d.get('_model') or d.get('model') or 'model'}:"
            f"{d.get('verdict')}/{d.get('recommendation')}"
            for d in dissenters
        ]
        dissent_summary = "; ".join(bits)

    # Prefer a suggested action from reconsider camp, else any allowed suggestion.
    suggested = ""
    for o in reconsider_votes + cleaned:
        family = str(o.get("suggested_action_family") or "")
        if family in ALLOWED_ACTIONS:
            suggested = family
            break

    citations: list[str] = []
    for o in cleaned:
        for c in o.get("evidence_citations") or []:
            text = str(c)
            if text and text not in citations:
                citations.append(text)

    risk_flags: list[str] = []
    for o in cleaned:
        for flag in o.get("risk_flags") or []:
            text = str(flag)
            if text and text not in risk_flags:
                risk_flags.append(text)

    qualities = [str(o.get("lesson_quality") or "adequate") for o in cleaned]
    if "weak" in qualities and recommendation == "reconsider":
        lesson_quality = "weak"
    elif qualities.count("strong") >= max(1, len(qualities) // 2 + 1):
        lesson_quality = "strong"
    else:
        lesson_quality = "adequate"

    summaries = [
        str(o.get("reasoning_summary") or "").strip()
        for o in camp
        if str(o.get("reasoning_summary") or "").strip()
    ]
    reasoning_summary = summaries[0] if summaries else "Panel consensus formed."

    alts = [
        str(o.get("alternative_explanation") or "").strip()
        for o in cleaned
        if str(o.get("alternative_explanation") or "").strip()
    ]

    merged = {
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "alternative_explanation": alts[0] if alts else "",
        "risk_flags": risk_flags,
        "recommendation": recommendation,
        "reasoning_summary": reasoning_summary,
        "evidence_citations": citations[:8],
        "suggested_action_family": suggested,
        "suggested_action_allowed": bool(suggested),
        "lesson_quality": lesson_quality,
        "panel": panel_rows,
        "agreement_rate": round(agreement_rate, 4),
        "dissent_summary": dissent_summary,
    }
    return normalize_judge(merged)
