"""Preference-pair construction and eligibility checks for honest demos."""

from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

import config
from src.eval.behavioral import ACTIONABLE_HARM_CUES, REFUSAL_CUES, analyze_behavior, _hit
from src.judge.schemas import PreferencePair

MIN_SAVE_CONFIDENCE = config.JUDGE_MIN_CONFIDENCE

COSMETIC_SUFFIX_RE = re.compile(
    r"(\n\s*)?\[(Improved|Revised|Better|Updated)[^\]]*\]\s*$",
    flags=re.I,
)

META_PREFERRED_RE = re.compile(
    r"\b(the response|the preferred response|preferred response should|"
    r"the assistant|the target|this reply|overall assessment|"
    r"the model|fails to provide|is safe and|is helpful|does not adhere|"
    r"providing a short example|without any violations)\b",
    flags=re.I,
)


def strip_cosmetic_suffix(text: str) -> str:
    return COSMETIC_SUFFIX_RE.sub("", (text or "").strip()).strip()


def normalized_overlap_ratio(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z0-9]+", (a or "").lower()))
    tb = set(re.findall(r"[a-z0-9]+", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def looks_like_meta_judgment(text: str) -> bool:
    """Detect preferred text that critiques the reply instead of answering as the assistant."""
    t = (text or "").strip()
    if not t:
        return True
    low = t.lower()
    if low.startswith("the response") or low.startswith("the preferred"):
        return True
    if META_PREFERRED_RE.search(t):
        # Even refusals that are actually meta-commentary about the reply
        if "the response" in low or "preferred response" in low or "the assistant" in low:
            return True
        if not _hit(REFUSAL_CUES, t):
            return True
    return False


def is_cosmetic_pair(chosen: str, rejected: str) -> bool:
    c = strip_cosmetic_suffix(chosen)
    r = (rejected or "").strip()
    if not c or not r:
        return True
    if c == r:
        return True
    # Tiny edit distance via overlap
    if normalized_overlap_ratio(c, r) >= 0.92 and abs(len(c) - len(r)) < max(24, int(0.15 * max(len(c), len(r)))):
        return True
    # Chosen is rejected plus a short boilerplate clause
    if r and c.startswith(r) and len(c) - len(r) < 80:
        return True
    return False


def chosen_teaches_harm(prompt: str, chosen: str) -> bool:
    """Reject preferences where the *chosen* answer still enables harm."""
    behavior = analyze_behavior(prompt, chosen)
    if behavior.get("under_refusal"):
        return True
    if behavior.get("intent_class") == "harmful" and _hit(ACTIONABLE_HARM_CUES, chosen):
        return True
    # If prompt is harmful and chosen does not refuse, treat as risky
    if behavior.get("intent_class") == "harmful" and not _hit(REFUSAL_CUES, chosen):
        # Allow short redirects that still refuse with softer language
        if len(chosen) > 40 and not any(
            w in chosen.lower() for w in ["can't", "cannot", "won't", "illegal", "not able"]
        ):
            return True
    return False


def assess_preference_eligibility(
    *,
    prompt: str,
    chosen: str,
    rejected: str,
    confidence: float,
    judgment_mode: str,
    verification_ok: bool,
    verification_issues: list[str] | None = None,
    source: str | None = None,
    rationale: str | None = None,
    retrieved_principle_ids: list[str] | None = None,
    require_principles: bool = True,
    behavior_hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Decide whether a preference pair may be saved / used for training.
    Low-confidence / malformed / heuristic / cosmetic / harmful-chosen are blocked.

    `behavior_hints` (optional) are the declared scenario expectations used by the
    agentic loop, so the "target already adequate" gate sees the same evidence.
    """
    issues: list[str] = []
    warnings: list[str] = []

    if not prompt.strip() or not chosen.strip() or not rejected.strip():
        issues.append("Missing prompt, chosen, or rejected text")
    if chosen.strip() == rejected.strip():
        issues.append("Chosen and rejected responses are identical")
    if is_cosmetic_pair(chosen, rejected):
        issues.append("Chosen/rejected pair is cosmetic (near-duplicate or suffix-only edit)")
    if looks_like_meta_judgment(chosen):
        issues.append("Chosen looks like meta-evaluation of the reply, not an assistant answer")
    # If the rejected/target already succeeded, DPO signal is weak
    rej_behavior = analyze_behavior(prompt, rejected, behavior_hints)
    rej_flags = set(rej_behavior.get("flags") or [])
    bad = {
        "over_refusal",
        "under_refusal_actionable_harm",
        "possible_fabrication",
        "format_violation",
    }
    if (rej_flags & {"appropriate_refusal", "appropriate_helpfulness"}) and not (rej_flags & bad):
        if not chosen_teaches_harm(prompt, chosen):
            # Only keep if chosen clearly repairs a remaining failure; otherwise skip
            cho_behavior = analyze_behavior(prompt, chosen, behavior_hints)
            if not (set(cho_behavior.get("flags") or []) & bad) and not looks_like_meta_judgment(
                chosen
            ):
                issues.append(
                    "Target response already adequate; preference adds little DPO signal"
                )
    if confidence < MIN_SAVE_CONFIDENCE:
        issues.append(
            f"Judge confidence {confidence:.2f} is below save threshold {MIN_SAVE_CONFIDENCE:.2f}"
        )
    if judgment_mode == "heuristic_fallback":
        issues.append("Heuristic fallback judgments are not eligible for training")
    if judgment_mode == "hybrid":
        warnings.append("Hybrid judgment — usable but lower epistemic quality than pure LLM")
    if not verification_ok:
        issues.append(
            "Verification failed: " + "; ".join(verification_issues or ["unknown issue"])
        )
    src = (source or "").lower()
    if src in {"judge_fallback", "smoke_seed", "manual_seed"}:
        issues.append(f"Source '{source}' is not trusted for training without re-judgment")
    if chosen_teaches_harm(prompt, chosen):
        issues.append("Chosen response still enables or teaches harmful behavior")
    if require_principles and not (retrieved_principle_ids or []):
        issues.append("No retrieved principle ids — preference is not retrieval-grounded")
    if not (rationale or "").strip() or len((rationale or "").strip()) < 20:
        issues.append("Missing or too-short rationale/critique")

    # Signal strength: chosen should differ meaningfully for DPO
    if normalized_overlap_ratio(chosen, rejected) > 0.88:
        issues.append("Insufficient contrast between chosen and rejected for DPO")

    eligible = len(issues) == 0
    reason = None if eligible else "; ".join(issues)
    return {
        "eligible": eligible,
        "reason": reason,
        "issues": issues,
        "warnings": warnings,
        "min_confidence": MIN_SAVE_CONFIDENCE,
        "training_status": "trusted" if eligible else "rejected_candidate",
    }


def assess_stored_preference_for_training(pair: dict[str, Any]) -> dict[str, Any]:
    """Re-evaluate an already-stored preference for training ingestion."""
    status = (pair.get("training_status") or "").lower()
    if status == "quarantined":
        return {
            "eligible": False,
            "reason": f"Quarantined: {pair.get('quarantine_reason') or 'legacy/contaminated'}",
            "issues": ["quarantined"],
            "warnings": [],
            "training_status": "quarantined",
        }
    if status == "trusted":
        # Still re-check critical safety properties; skip soft metadata re-litigation
        hard_issues = []
        if chosen_teaches_harm(pair.get("prompt", ""), pair.get("chosen", "")):
            hard_issues.append("chosen_teaches_harm")
        if is_cosmetic_pair(pair.get("chosen", ""), pair.get("rejected", "")):
            hard_issues.append("cosmetic_pair")
        if (pair.get("chosen") or "").strip() == (pair.get("rejected") or "").strip():
            hard_issues.append("identical_pair")
        if hard_issues:
            return {
                "eligible": False,
                "reason": "Trusted flag overridden: " + ", ".join(hard_issues),
                "issues": hard_issues,
                "warnings": [],
                "training_status": "quarantined",
            }
        return {
            "eligible": True,
            "reason": None,
            "issues": [],
            "warnings": [],
            "training_status": "trusted",
        }

    return assess_preference_eligibility(
        prompt=str(pair.get("prompt") or ""),
        chosen=str(pair.get("chosen") or ""),
        rejected=str(pair.get("rejected") or ""),
        confidence=float(pair.get("confidence") or 0.0),
        judgment_mode=str(pair.get("judgment_mode") or ""),
        verification_ok=True,  # historical rows may lack verification payload
        verification_issues=[],
        source=str(pair.get("source") or ""),
        rationale=str(pair.get("rationale") or pair.get("critique") or ""),
        retrieved_principle_ids=list(pair.get("retrieved_principle_ids") or []),
        require_principles=True,
    )


def build_preference_pair(
    *,
    prompt: str,
    chosen: str,
    rejected: str,
    judge_result_id: str | None,
    scores: dict[str, Any],
    overall_score: float | None,
    confidence: float | None,
    category: str | None,
    source: str,
    critique: str,
    score_explanation: str,
    retrieved_principle_ids: list[str],
    retrieved_principles: list[dict[str, Any]] | None,
    judgment_mode: str,
    judge_model: str | None,
    target_model: str | None = None,
    target_backend: str | None = None,
    training_status: str | None = None,
) -> dict[str, Any]:
    pair = PreferencePair(
        id=str(uuid4()),
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        judge_result_id=judge_result_id,
        scores=scores,
        overall_score=overall_score,
        confidence=confidence,
        category=category,
        source=source,
        rationale=critique,
        score_explanation=score_explanation,
        retrieved_principle_ids=list(retrieved_principle_ids or []),
        retrieved_references=[
            {
                "id": p.get("id"),
                "title": p.get("title"),
                "category": p.get("category"),
                "score": p.get("score"),
                "backend": p.get("backend"),
                "doc_type": p.get("doc_type"),
            }
            for p in (retrieved_principles or [])
        ],
        judgment_mode=judgment_mode,
        judge_model=judge_model,
        target_model=target_model,
        target_backend=target_backend,
    )
    data = pair.model_dump()
    data["training_status"] = training_status or "candidate"
    return data
