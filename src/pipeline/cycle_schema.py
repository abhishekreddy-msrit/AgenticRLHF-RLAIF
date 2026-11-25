"""
Stable result schema for the agentic feedback cycle.

Why this exists: the agentic loop and the classic V0.4 JudgeAgent return
differently shaped dictionaries. Rendering code that indexes keys directly
(`res["score_explanation"]`) crashes when handed the other shape. Everything
the UI renders goes through `normalize_cycle` / `normalize_judgment` /
`normalize_legacy_judgment`, which guarantee every key exists with an honest
default (None / [] / "not produced") rather than an invented value.

No Streamlit imports here — these functions are unit-testable.
"""

from __future__ import annotations

import json
from typing import Any

CYCLE_SCHEMA_VERSION = "0.5.1"

STAGES: list[tuple[str, str]] = [
    ("input", "User input"),
    ("target_original", "Target response"),
    ("investigation", "Agent investigation"),
    ("retrieved_context", "Retrieved context"),
    ("feedback", "Supervisory feedback"),
    ("revision", "Target revision"),
    ("verification", "Agent verification"),
    ("preference", "Training implication"),
]
STAGE_KEYS = [k for k, _ in STAGES]
STAGE_STATUSES = {"done", "skipped", "failed", "not_reached"}

VERIFICATION_STATUSES = {
    "corrected",
    "improved",
    "unchanged",
    "regressed",
    "inconclusive",
    "no_revision_needed",
    "not_run",
}

AUTONOMY_LABELS = {
    "agentic": "Agentic — the LLM chose each step",
    "agentic_degraded": "Agentic (degraded) — the LLM planned its tools once, then judged",
    "fixed_workflow_fallback": "Fixed workflow fallback — NOT autonomous",
    "not_run": "Not run",
}


def _as_dict(x: Any) -> dict:
    return x if isinstance(x, dict) else {}


def _as_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _as_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def _as_float(x: Any) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


# --------------------------------------------------------------------------- #
# Agentic judgment
# --------------------------------------------------------------------------- #
def normalize_judgment(j: Any) -> dict[str, Any] | None:
    """Fill every key the UI reads from a SupervisoryAgent judgment. None stays None."""
    if not isinstance(j, dict) or not j:
        return None
    scores = j.get("scores") if isinstance(j.get("scores"), dict) else {}
    return {
        "autonomy_mode": str(j.get("autonomy_mode") or "not_run"),
        "judgment_mode": j.get("judgment_mode"),
        "judge_model": j.get("judge_model"),
        "phase": j.get("phase"),
        "degraded_reason": j.get("degraded_reason"),
        "fallback_reason": j.get("fallback_reason"),
        "action_trace": [t for t in _as_list(j.get("action_trace")) if isinstance(t, dict)],
        "tool_observations": [o for o in _as_list(j.get("tool_observations")) if isinstance(o, dict)],
        "tools_available": [str(t) for t in _as_list(j.get("tools_available"))],
        "tool_calls_count": int(j.get("tool_calls_count") or 0),
        "scores": scores,
        "scores_reported_by_agent": bool(j.get("scores_reported_by_agent", bool(scores))),
        "overall_score": _as_float(j.get("overall_score")),
        "issues_found": [str(x) for x in _as_list(j.get("issues_found"))],
        "agent_issues": [str(x) for x in _as_list(j.get("agent_issues"))],
        "rule_issues": [str(x) for x in _as_list(j.get("rule_issues"))],
        "critique": _as_text(j.get("critique")),
        "overall_assessment": _as_text(j.get("overall_assessment")),
        "feedback_to_target": _as_text(j.get("feedback_to_target")),
        "agent_feedback": _as_text(j.get("agent_feedback")),
        "rule_addenda": [str(x) for x in _as_list(j.get("rule_addenda"))],
        "agent_feedback_vetoed": j.get("agent_feedback_vetoed"),
        "feedback_source": str(j.get("feedback_source") or "unknown"),
        "confidence": _as_float(j.get("confidence")),
        "confidence_reported_by_agent": bool(j.get("confidence_reported_by_agent", False)),
        "confidence_caps": [str(x) for x in _as_list(j.get("confidence_caps"))],
        "retrieved_principles": [p for p in _as_list(j.get("retrieved_principles")) if isinstance(p, dict)],
        "retrieved_memory": [m for m in _as_list(j.get("retrieved_memory")) if isinstance(m, dict)],
        "agent_reported_evidence": [str(x) for x in _as_list(j.get("agent_reported_evidence"))],
        "behavioral_precheck": _as_dict(j.get("behavioral_precheck")),
        "rule_repair_notes": [str(x) for x in _as_list(j.get("rule_repair_notes"))],
        "why_finished": j.get("why_finished"),
        "feedback_followed_agent": j.get("feedback_followed_agent"),
    }


# --------------------------------------------------------------------------- #
# Cycle
# --------------------------------------------------------------------------- #
def new_cycle(prompt: str = "", category: str | None = None) -> dict[str, Any]:
    """An empty, fully-populated cycle. The loop fills it in stage by stage."""
    return {
        "schema_version": CYCLE_SCHEMA_VERSION,
        "status": "running",  # running | completed | completed_no_revision | failed
        "error": None,  # {stage, stage_label, type, message, traceback}
        "stages": {k: {"status": "not_reached", "seconds": None, "note": None} for k in STAGE_KEYS},
        "started_at": None,
        "total_seconds": None,
        "session_id": None,
        "db_path": None,
        # stage 1
        "prompt": prompt,
        "category": category,
        "scenario": None,
        "evaluation_focus": None,
        "hints": {},
        # stage 2
        "target_info": {},
        "initial_response": None,
        "initial_generation_meta": {},
        "initial_behavior": None,
        # stage 3 / 4
        "judge_model": None,
        "judge_info": {},
        "initial_judgment": None,
        # stage 5
        "revision_triggers": [],
        "feedback_to_target": None,
        "feedback": None,  # {text, source, agent_text, rule_addenda, vetoed_reason}
        "revision_prompt": None,
        # stage 6
        "revised_response": None,
        "revised_generation_meta": {},
        "revised_behavior": None,
        # stage 7
        "verify_judgment": None,
        "verification": {"status": "not_run", "genuinely_better": False},
        # stage 8
        "preference": None,
        "preference_decision": None,  # {created: bool, reason: str}
        "save_requested": False,
        "saved_preference_id": None,
        "save_blocked_reason": None,
        "weights_unchanged": True,
        "note": (
            "In-session behavioral correction only. Target weights were not updated. "
            "Eligible original→revised pairs may be stored for future DPO."
        ),
    }


def normalize_cycle(raw: Any) -> dict[str, Any]:
    """
    Coerce any cycle-like dict (current, older V0.5, partial, or failed) into the
    stable schema. Never raises; unknown input yields a clearly failed cycle.
    """
    if not isinstance(raw, dict):
        out = new_cycle()
        out["status"] = "failed"
        out["error"] = {
            "stage": None,
            "stage_label": None,
            "type": "InvalidResult",
            "message": f"Agentic cycle returned {type(raw).__name__} instead of a result object.",
            "traceback": None,
        }
        return out

    out = new_cycle(str(raw.get("prompt") or ""), raw.get("category"))
    for key in out:
        if key in raw and raw[key] is not None:
            out[key] = raw[key]

    out["initial_judgment"] = normalize_judgment(raw.get("initial_judgment"))
    out["verify_judgment"] = normalize_judgment(raw.get("verify_judgment"))
    out["target_info"] = _as_dict(raw.get("target_info"))
    out["judge_info"] = _as_dict(raw.get("judge_info"))
    out["hints"] = _as_dict(raw.get("hints"))
    out["scenario"] = raw.get("scenario") if isinstance(raw.get("scenario"), dict) else None
    out["revision_triggers"] = [str(x) for x in _as_list(raw.get("revision_triggers"))]
    out["initial_generation_meta"] = _as_dict(raw.get("initial_generation_meta"))
    out["revised_generation_meta"] = _as_dict(raw.get("revised_generation_meta"))

    ver = _as_dict(raw.get("verification"))
    status = str(ver.get("status") or "not_run")
    if status == "new_issue":  # legacy V0.5 label
        status = "regressed"
    if status not in VERIFICATION_STATUSES:
        status = "inconclusive"
    out["verification"] = {
        "status": status,
        "genuinely_better": bool(ver.get("genuinely_better", False)),
        "evidence": ver.get("evidence"),
        "reason": ver.get("reason") or ver.get("note"),
        "score_delta": _as_float(ver.get("score_delta")),
        "scores_comparable": bool(ver.get("scores_comparable", False)),
        "behavioral_before": _as_dict(ver.get("behavioral_before")),
        "behavioral_after": _as_dict(ver.get("behavioral_after")),
        "feedback_followed": ver.get("feedback_followed"),
        "feedback_followed_basis": ver.get("feedback_followed_basis"),
        "feedback_followed_agent": ver.get("feedback_followed_agent"),
        "fixed_flags": [str(x) for x in _as_list(ver.get("fixed_flags"))],
        "remaining_flags": [str(x) for x in _as_list(ver.get("remaining_flags"))],
        "new_flags": [str(x) for x in _as_list(ver.get("new_flags"))],
        "agent_issues_on_revision": [str(x) for x in _as_list(ver.get("agent_issues_on_revision"))],
        "caveats": [str(x) for x in _as_list(ver.get("caveats"))],
        "changes": _as_dict(ver.get("changes")),
        "note": ver.get("note"),
    }

    # Feedback card: tolerate legacy cycles that only carried a string
    fb = raw.get("feedback")
    text = raw.get("feedback_to_target")
    if isinstance(fb, dict):
        out["feedback"] = {
            "text": _as_text(fb.get("text") or text),
            "source": str(fb.get("source") or "unknown"),
            "agent_text": _as_text(fb.get("agent_text")),
            "rule_addenda": [str(x) for x in _as_list(fb.get("rule_addenda"))],
            "vetoed_reason": fb.get("vetoed_reason"),
        }
    elif text:
        out["feedback"] = {
            "text": _as_text(text),
            "source": "unknown",
            "agent_text": "",
            "rule_addenda": [],
            "vetoed_reason": None,
        }
    else:
        out["feedback"] = None

    pref = raw.get("preference")
    if isinstance(pref, dict):
        out["preference"] = {
            "pair": _as_dict(pref.get("pair")),
            "eligibility": _as_dict(pref.get("eligibility")),
        }
    else:
        out["preference"] = None

    # Stage table: tolerate legacy cycles with no stage info by inferring it
    stages_in = _as_dict(raw.get("stages"))
    stages: dict[str, dict[str, Any]] = {}
    for key in STAGE_KEYS:
        s = _as_dict(stages_in.get(key))
        st = s.get("status")
        if st not in STAGE_STATUSES:
            st = _infer_stage_status(key, out)
        stages[key] = {"status": st, "seconds": _as_float(s.get("seconds")), "note": s.get("note")}
    out["stages"] = stages

    err = raw.get("error")
    if isinstance(err, dict) and (err.get("message") or err.get("type")):
        out["error"] = {
            "stage": err.get("stage"),
            "stage_label": err.get("stage_label") or dict(STAGES).get(err.get("stage"), err.get("stage")),
            "type": str(err.get("type") or "Error"),
            "message": _as_text(err.get("message")),
            "traceback": err.get("traceback"),
        }
        out["status"] = "failed"
    else:
        out["error"] = None
        if out["status"] not in {"completed", "completed_no_revision", "failed", "running"}:
            out["status"] = "completed"
        if "status" not in raw:  # legacy cycle without explicit status
            out["status"] = "completed" if out["revised_response"] else "completed_no_revision"
    return out


def _infer_stage_status(key: str, c: dict[str, Any]) -> str:
    j0 = c.get("initial_judgment") or {}
    if key == "input":
        return "done" if c.get("prompt") else "not_reached"
    if key == "target_original":
        return "done" if c.get("initial_response") else "not_reached"
    if key == "investigation":
        return "done" if j0 else "not_reached"
    if key == "retrieved_context":
        if not j0:
            return "not_reached"
        return "done" if (j0.get("retrieved_principles") or j0.get("retrieved_memory")) else "skipped"
    if key == "feedback":
        if not j0:
            return "not_reached"
        return "done" if c.get("feedback_to_target") else "skipped"
    if key == "revision":
        if not j0:
            return "not_reached"
        return "done" if c.get("revised_response") else "skipped"
    if key == "verification":
        if not j0:
            return "not_reached"
        return "done" if c.get("verify_judgment") else "skipped"
    if key == "preference":
        return "done" if j0 else "not_reached"
    return "not_reached"


# --------------------------------------------------------------------------- #
# Timeline (stage 3): built ONLY from real trace entries and real tool results
# --------------------------------------------------------------------------- #
def build_timeline(judgment: dict[str, Any] | None) -> list[dict[str, Any]]:
    """
    Turn the raw action trace into ordered cards of four honest kinds:
      agent_decision | tool_execution | agent_conclusion | harness_note
    Tool cards carry the full real tool result (looked up by call_index).
    Nothing is synthesized: a card exists only if a trace entry exists.
    """
    j = normalize_judgment(judgment)
    if not j:
        return []
    obs_by_index = {o.get("call_index"): o for o in j["tool_observations"]}
    cards: list[dict[str, Any]] = []

    def add(**card: Any) -> None:
        card["n"] = len(cards) + 1
        cards.append(card)

    for entry in j["action_trace"]:
        ev = entry.get("event")
        d = entry.get("detail")
        dd = d if isinstance(d, dict) else {}
        if ev == "agent_decision":
            if dd.get("decision") == "use_tool":
                add(
                    kind="agent_decision",
                    title=f"Agent decided to call `{dd.get('tool')}`",
                    text=str(dd.get("why") or "No reason given by the agent."),
                    tool=dd.get("tool"),
                    args=dd.get("args") or {},
                    step=dd.get("step"),
                    planned=bool(dd.get("planned")),
                )
            elif dd.get("decision") == "finish":
                n_tools = dd.get("tools_used_before_finish")
                add(
                    kind="agent_conclusion",
                    title=(
                        "Agent decided it had enough evidence and finished"
                        if n_tools
                        else "Agent finished without calling any tool"
                    ),
                    text=str(dd.get("why") or "No reason given by the agent."),
                    step=dd.get("step"),
                    tools_used=n_tools,
                )
        elif ev == "tool_result":
            obs = obs_by_index.get(dd.get("call_index")) or {}
            add(
                kind="tool_execution",
                title=f"Tool `{dd.get('tool')}` executed"
                + ("" if dd.get("ok") else " — returned an error"),
                tool=dd.get("tool"),
                ok=bool(dd.get("ok")),
                args=obs.get("args") or {},
                result=obs.get("result"),
                error=dd.get("error") or obs.get("error"),
                text=summarize_tool_result(obs) if obs else "",
            )
        elif ev == "tool_plan":
            tools = dd.get("tools") or []
            add(
                kind="agent_decision",
                title="Agent planned its tools in one step"
                + (f": {', '.join(f'`{t}`' for t in tools)}" if tools else ": no tools"),
                text=str(dd.get("focus") or ""),
                planned=True,
            )
        elif ev == "implicit_finish":
            continue  # the following agent_decision(finish) card covers it
        elif ev == "degraded_plan_mode":
            add(
                kind="harness_note",
                level="warn",
                title="Harness switched to degraded mode (plan once, then judge)",
                text=f"Reason: {dd.get('reason') or d}",
            )
        elif ev == "fixed_workflow_fallback":
            add(
                kind="harness_note",
                level="error",
                title="Harness fell back to the fixed judge workflow — not autonomous",
                text=f"Reason: {dd.get('reason') or d}",
            )
        elif ev == "decision_parse_error":
            add(
                kind="harness_note",
                level="warn",
                title="Agent's decision could not be parsed as JSON",
                text=str(dd.get("error") or d),
            )
        elif ev == "max_steps_reached":
            add(
                kind="harness_note",
                level="warn",
                title=f"Step limit reached ({dd.get('max_steps') or d} steps) without a final judgment",
                text="",
            )
        elif ev == "unknown_decision":
            add(
                kind="harness_note",
                level="warn",
                title="Agent returned an unrecognized decision; harness asked it to retry",
                text=f"Keys returned: {dd.get('keys')}",
            )
        elif ev == "rule_score_repair":
            add(
                kind="harness_note",
                level="info",
                title="Rule verifier corrected the agent's scores (deterministic, not the LLM)",
                text="; ".join(str(x) for x in _as_list(d)),
            )
        elif ev == "feedback_rule_addendum":
            add(
                kind="harness_note",
                level="info",
                title="Rule verifier added factual lines to the feedback (deterministic, not the LLM)",
                text=" ".join(str(x) for x in _as_list(d)),
            )
        elif ev == "agent_feedback_vetoed":
            add(
                kind="harness_note",
                level="warn",
                title="Harness did not forward the agent's feedback text",
                text=f"Because {d}.",
            )
        elif ev == "scores_missing":
            add(kind="harness_note", level="warn", title="Agent returned no parsable scores", text=str(d))
        elif ev in {"agent_unavailable", "degraded_plan_failed", "degraded_judge_failed"}:
            add(
                kind="harness_note",
                level="error",
                title=str(ev).replace("_", " ").capitalize(),
                text=_as_text(dd.get("error") or d),
            )
        # Any other bookkeeping event is intentionally not shown as a step.
    return cards


def summarize_tool_result(obs: dict[str, Any]) -> str:
    """One factual sentence about what a tool returned (no interpretation)."""
    if not obs.get("ok"):
        return f"Error: {obs.get('error')}"
    tool = obs.get("tool")
    res = obs.get("result")
    r = res if isinstance(res, dict) else {}
    if tool == "retrieve_safety_principles":
        hits = r.get("hits") or []
        return f"Returned {len(hits)} document(s) from the {r.get('backend') or 'vector'} store."
    if tool == "analyze_behavior":
        flags = r.get("flags") or []
        return "Flags: " + (", ".join(flags) if flags else "none") + f" · intent class: {r.get('intent_class')}"
    if tool == "get_rubric":
        dims = r.get("dimensions") or []
        return f"Returned the rubric ({len(dims)} dimensions)."
    if tool == "read_target_system_prompt":
        return "Returned the target's system prompt (read-only)."
    if tool == "inspect_conversation_history":
        return f"Returned {len(r.get('turns') or [])} conversation turn(s) from this cycle."
    if tool == "retrieve_prior_evaluations":
        return f"Returned {len(r.get('evaluations') or [])} prior evaluation(s) from memory."
    return "Tool returned a result."


def tool_usage(judgment: dict[str, Any] | None) -> dict[str, Any]:
    """Which available tools the agent used / did not use in this investigation."""
    j = normalize_judgment(judgment)
    if not j:
        return {"available": [], "used": [], "not_used": [], "used_by_other": []}
    used_agent: list[str] = []
    used_other: list[str] = []
    for o in j["tool_observations"]:
        name = str(o.get("tool"))
        if o.get("selected_by") == "agent":
            if name not in used_agent:
                used_agent.append(name)
        elif name not in used_other:
            used_other.append(name)
    available = j["tools_available"]
    return {
        "available": available,
        "used": used_agent,
        "not_used": [t for t in available if t not in used_agent and t not in used_other],
        "used_by_other": used_other,
    }


def principle_citation(principle: dict[str, Any], judgment: dict[str, Any] | None) -> str:
    """
    How a retrieved document relates to the agent's judgment, stated conservatively:
      "named_by_agent"  — the agent listed its id/title in evidence_used or quoted it in its text
      "returned_only"   — it was returned to the agent; influence on the judgment is not measurable
    """
    j = normalize_judgment(judgment)
    if not j:
        return "returned_only"
    pid = str(principle.get("id") or "").lower()
    title = str(principle.get("title") or "").lower()
    haystack = " ".join(
        [j["critique"], j["agent_feedback"], j["overall_assessment"], " ".join(j["agent_reported_evidence"])]
    ).lower()
    if pid and pid in haystack:
        return "named_by_agent"
    if title and len(title) >= 8 and title in haystack:
        return "named_by_agent"
    return "returned_only"


# --------------------------------------------------------------------------- #
# Classic (V0.4) JudgeAgent payload — used by the Judge tab
# --------------------------------------------------------------------------- #
LEGACY_RESULT_TEXT_FIELDS = (
    "critique",
    "score_explanation",
    "overall_assessment",
    "preferred_response",
    "rejected_response",
    "target_response",
)


def normalize_legacy_judgment(j: Any) -> dict[str, Any] | None:
    """
    Make a classic JudgeAgent payload safe to render. Missing fields become None/[]
    and are listed in `missing_fields` so the UI can say "not produced" instead of
    crashing or showing an invented value.
    """
    if not isinstance(j, dict) or not j:
        return None
    res_in = j.get("result") if isinstance(j.get("result"), dict) else {}
    missing: list[str] = []
    if not isinstance(j.get("result"), dict):
        missing.append("result")

    res: dict[str, Any] = dict(res_in)
    for f in LEGACY_RESULT_TEXT_FIELDS:
        v = res_in.get(f)
        if v is None or (isinstance(v, str) and not v.strip()):
            res[f] = None
            missing.append(f"result.{f}")
        else:
            res[f] = _as_text(v)
    scores = res_in.get("scores") if isinstance(res_in.get("scores"), dict) else {}
    if not scores:
        missing.append("result.scores")
    res["scores"] = scores
    res["overall_score"] = _as_float(res_in.get("overall_score"))
    if res["overall_score"] is None:
        missing.append("result.overall_score")
    res["confidence"] = _as_float(res_in.get("confidence"))
    if res["confidence"] is None:
        missing.append("result.confidence")
    res["violations"] = [str(x) for x in _as_list(res_in.get("violations"))]

    ver_in = j.get("verification")
    if isinstance(ver_in, dict):
        verification = {
            "present": True,
            "ok": bool(ver_in.get("ok")),
            "issues": [str(x) for x in _as_list(ver_in.get("issues"))],
        }
    else:
        verification = {"present": False, "ok": False, "issues": []}
        missing.append("verification")

    tool_calls = []
    for c in _as_list(j.get("tool_calls")):
        if isinstance(c, dict):
            tool_calls.append(
                {"tool": str(c.get("tool") or "unknown"), "args": c.get("args"), "result_preview": c.get("result_preview")}
            )

    return {
        "result": res,
        "judgment_mode": j.get("judgment_mode"),
        "used_fallback": bool(j.get("used_fallback")),
        "judge_model": j.get("judge_model"),
        "retrieval_status": _as_dict(j.get("retrieval_status")),
        "retrieved_count": int(j.get("retrieved_count") or 0),
        "retrieved_principles": [p for p in _as_list(j.get("retrieved_principles")) if isinstance(p, dict)],
        "influential_principles": [p for p in _as_list(j.get("influential_principles")) if isinstance(p, dict)],
        "tool_calls": tool_calls,
        "preference_eligibility": _as_dict(j.get("preference_eligibility")),
        "preference_pair": j.get("preference_pair") if isinstance(j.get("preference_pair"), dict) else None,
        "verification": verification,
        "missing_fields": missing,
    }
