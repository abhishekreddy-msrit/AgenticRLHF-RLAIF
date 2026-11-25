"""
Streamlit rendering for the V0.5 Agentic Feedback Loop.

Renders only through the stable cycle schema (normalize_cycle). Missing optional
fields become "not produced" — never KeyError, never invented success.
"""

from __future__ import annotations

import html
import json
from typing import Any

import streamlit as st

import config
from src.pipeline.cycle_schema import (
    AUTONOMY_LABELS,
    STAGES,
    build_timeline,
    normalize_cycle,
    principle_citation,
    tool_usage,
)


SCENARIO_PATH = config.BENCHMARK_DIR / "agentic_demo_scenarios.json"


def load_agentic_scenarios() -> list[dict]:
    data = json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))
    return list(data.get("scenarios") or [])


def _esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def _fmt_score(x: Any) -> str:
    try:
        return f"{float(x):.2f}"
    except (TypeError, ValueError):
        return "not produced"


def _status_class(status: str) -> str:
    return {
        "corrected": "ok",
        "improved": "ok",
        "unchanged": "warn",
        "regressed": "fallback",
        "inconclusive": "warn",
        "no_revision_needed": "llm",
        "not_run": "warn",
        "failed": "fallback",
        "done": "ok",
        "skipped": "warn",
        "running": "hybrid",
        "completed": "ok",
        "completed_no_revision": "llm",
    }.get(status, "")


def render_stage_rail(cycle: dict[str, Any]) -> None:
    stages = cycle.get("stages") or {}
    chips = []
    for key, label in STAGES:
        stt = (stages.get(key) or {}).get("status", "not_reached")
        sec = (stages.get(key) or {}).get("seconds")
        sec_s = f" · {sec:.1f}s" if isinstance(sec, (int, float)) else ""
        chips.append(
            f'<span class="stage-chip {_status_class(stt)}">'
            f'<span class="stage-dot"></span>{_esc(label)}'
            f'<span class="stage-state">{_esc(stt)}{sec_s}</span></span>'
        )
    st.markdown(f'<div class="stage-rail">{"".join(chips)}</div>', unsafe_allow_html=True)


def render_summary(cycle: dict[str, Any]) -> None:
    j0 = cycle.get("initial_judgment") or {}
    ver = cycle.get("verification") or {}
    autonomy = j0.get("autonomy_mode") or "not_run"
    target = (cycle.get("target_info") or {}).get("model_id") or "unknown"
    judge = cycle.get("judge_model") or (cycle.get("judge_info") or {}).get("active_model") or "unknown"
    status = cycle.get("status") or "unknown"
    st.markdown(
        f"""
        <div class="panel summary-panel">
          <div class="summary-row">
            <span class="badge {_status_class(status)}">Cycle: {_esc(status)}</span>
            <span class="badge {_status_class(autonomy) if autonomy != 'fixed_workflow_fallback' else 'fallback'}">{_esc(AUTONOMY_LABELS.get(autonomy, autonomy))}</span>
            <span class="badge">Target · {_esc(target)}</span>
            <span class="badge">Judge · {_esc(judge)}</span>
            <span class="badge {_status_class(ver.get('status') or 'not_run')}">Verify · {_esc(ver.get('status') or 'not_run')}</span>
          </div>
          <p class="soft" style="margin:0.85rem 0 0;color:#3a3a3c !important;">
            {_esc(cycle.get("note") or "")}
            {f' · total {cycle.get("total_seconds")}s' if cycle.get("total_seconds") is not None else ""}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_error(cycle: dict[str, Any]) -> None:
    err = cycle.get("error")
    if not err:
        return
    st.error(
        f"**Pipeline failed at stage `{err.get('stage_label') or err.get('stage')}`** — "
        f"{err.get('type')}: {err.get('message')}"
    )
    tb = err.get("traceback")
    if tb:
        with st.expander("Diagnostics / traceback", expanded=False):
            st.code(tb)


def _response_card(title: str, body: str | None, meta: str = "") -> None:
    text = (body or "").strip()
    if not text:
        st.markdown(
            f'<div class="panel"><h3>{_esc(title)}</h3>'
            f'<p class="soft" style="color:#3a3a3c !important;">Not produced.</p></div>',
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        f'<div class="panel"><h3>{_esc(title)}</h3>'
        f'{f"<p class=soft style=color:#3a3a3c!important;>{_esc(meta)}</p>" if meta else ""}'
        f'<div class="resp">{_esc(text)}</div></div>',
        unsafe_allow_html=True,
    )


def render_timeline(
    judgment: dict | None, title: str, *, allow_expanders: bool = True
) -> None:
    cards = build_timeline(judgment)
    st.markdown(f"#### {title}")
    if not cards:
        st.info("No investigation steps were recorded.")
        return
    usage = tool_usage(judgment)
    if usage.get("available"):
        st.caption(
            "Tools available: "
            + ", ".join(f"`{t}`" for t in usage["available"])
            + (" · used by agent: " + ", ".join(f"`{t}`" for t in usage["used"]) if usage["used"] else " · used by agent: none")
            + (
                " · not used: " + ", ".join(f"`{t}`" for t in usage["not_used"])
                if usage.get("not_used")
                else ""
            )
        )
    for card in cards:
        kind = card.get("kind")
        n = card.get("n")
        title_c = card.get("title") or ""
        text = card.get("text") or ""
        level = card.get("level") or ""
        css = {
            "agent_decision": "trace-decision",
            "tool_execution": "trace-tool",
            "agent_conclusion": "trace-finish",
            "harness_note": "trace-note",
        }.get(kind, "trace-note")
        if level == "error":
            css += " trace-error"
        elif level == "warn":
            css += " trace-warn"
        st.markdown(
            f'<div class="trace-card {css}"><div class="trace-n">{n}</div>'
            f'<div class="trace-body"><div class="trace-title">{_esc(title_c)}</div>'
            f'<div class="trace-text">{_esc(text)}</div></div></div>',
            unsafe_allow_html=True,
        )
        if kind == "tool_execution":
            label = f"Step {n} · full tool result (`{card.get('tool')}`)"

            def _tool_body() -> None:
                if card.get("args"):
                    st.markdown("**Args**")
                    st.json(card.get("args"))
                if card.get("error"):
                    st.error(card.get("error"))
                elif card.get("result") is not None:
                    st.json(card.get("result"))
                else:
                    st.caption("No result payload stored.")

            if allow_expanders:
                with st.expander(label, expanded=False):
                    _tool_body()
            else:
                st.markdown(f"**{label}**")
                _tool_body()


def render_retrieval(judgment: dict | None) -> None:
    j = judgment or {}
    principles = j.get("retrieved_principles") or []
    memory = j.get("retrieved_memory") or []
    if not principles and not memory:
        st.info("No retrieval/memory tools returned documents in this investigation.")
        return
    if principles:
        st.markdown("**Safety principles / examples from the vector store**")
        for p in principles:
            cite = principle_citation(p, j)
            cite_label = (
                "Named / referenced in the agent's written judgment"
                if cite == "named_by_agent"
                else "Returned to the agent — influence on the judgment is not measurable"
            )
            badge = "ok" if cite == "named_by_agent" else "warn"
            with st.expander(
                f"{p.get('title') or p.get('id')} · {p.get('category') or ''} · "
                f"score={p.get('score')} · {_esc(p.get('doc_type') or '')}"
            ):
                st.markdown(
                    f'<span class="badge {badge}">{_esc(cite_label)}</span>',
                    unsafe_allow_html=True,
                )
                st.write(p.get("excerpt") or p.get("text") or p.get("principle") or "(empty)")
                st.caption(
                    f"id={p.get('id')} · backend={p.get('backend')} · source_type={p.get('doc_type')}"
                )
    if memory:
        st.markdown("**Prior evaluations / memory**")
        for m in memory:
            with st.expander(f"Memory · {m.get('id') or m.get('title') or 'item'}"):
                st.json(m)


def render_cycle(result: dict[str, Any], *, scenario_meta: dict | None = None) -> None:
    """Main Agentic Loop report. Always safe against missing keys."""
    cycle = normalize_cycle(result)

    render_summary(cycle)
    render_stage_rail(cycle)
    render_error(cycle)

    # ---- STAGE 1 ----
    st.markdown('<div class="stage-header">Stage 1 — User input / test case</div>', unsafe_allow_html=True)
    scen = cycle.get("scenario") or scenario_meta or {}
    st.markdown(
        f"""
        <div class="panel">
          <p><strong>Scenario:</strong> {_esc(scen.get('title') or scen.get('id') or 'Custom prompt')}</p>
          <p><strong>Category / focus:</strong> {_esc(cycle.get('category') or 'n/a')}
             {(' · ' + _esc(cycle.get('evaluation_focus'))) if cycle.get('evaluation_focus') else ''}</p>
          <p><strong>Why this test:</strong> {_esc(scen.get('why_demo') or 'Operator-supplied custom prompt.')}</p>
          <p><strong>Success looks like:</strong> {_esc(scen.get('success_after_feedback') or 'n/a')}</p>
          <h3 style="margin-top:1rem;">Prompt sent to the target</h3>
          <div class="resp">{_esc(cycle.get('prompt'))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- STAGE 2 ----
    st.markdown(
        '<div class="stage-header">Stage 2 — Target LLM: original response</div>',
        unsafe_allow_html=True,
    )
    j0 = cycle.get("initial_judgment") or {}
    beh0 = cycle.get("initial_behavior") or {}
    meta0 = (
        f"Model: {(cycle.get('target_info') or {}).get('model_id') or 'unknown'} · "
        f"overall={_fmt_score(j0.get('overall_score'))} · "
        f"confidence={_fmt_score(j0.get('confidence'))} · "
        f"behavioral success={beh0.get('success')} · "
        f"flags={', '.join(beh0.get('flags') or beh0.get('bad_flags') or []) or 'none'}"
    )
    _response_card("Original target response", cycle.get("initial_response"), meta0)
    issues = j0.get("issues_found") or j0.get("agent_issues") or []
    if issues:
        st.markdown("**Issues detected on the original response**")
        for i in issues:
            st.write(f"- {i}")
    scores = j0.get("scores") or {}
    if scores:
        cols = st.columns(min(3, max(1, len(scores))))
        for i, (k, v) in enumerate(scores.items()):
            cols[i % len(cols)].metric(k.replace("_", " "), v)
    elif j0:
        st.caption("Scores: not produced by the agent for this run.")

    # ---- STAGE 3 ----
    st.markdown(
        '<div class="stage-header">Stage 3 — Supervisory agent investigation</div>',
        unsafe_allow_html=True,
    )
    if j0:
        render_timeline(j0, "Chronological agent activity (initial)")
        st.markdown(
            f'<div class="panel"><strong>Agent assessment:</strong> '
            f'{_esc(j0.get("overall_assessment") or "not produced")}<br/>'
            f'<strong>Critique:</strong> {_esc(j0.get("critique") or "not produced")}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("Investigation stage did not produce a judgment.")

    # ---- STAGE 4 ----
    st.markdown(
        '<div class="stage-header">Stage 4 — Retrieved context / memory</div>',
        unsafe_allow_html=True,
    )
    render_retrieval(j0)

    # ---- STAGE 5 ----
    st.markdown(
        '<div class="stage-header">Stage 5 — Supervisory feedback</div>',
        unsafe_allow_html=True,
    )
    fb = cycle.get("feedback") or {}
    fb_text = (fb.get("text") if isinstance(fb, dict) else None) or cycle.get("feedback_to_target")
    if fb_text:
        src = (fb.get("source") if isinstance(fb, dict) else None) or "unknown"
        st.markdown(
            f"""
            <div class="panel feedback-panel">
              <div class="badge hybrid">Feedback source · {_esc(src)}</div>
              <h3 style="margin-top:0.8rem;">Instructions sent to the target</h3>
              <div class="resp">{_esc(fb_text)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if isinstance(fb, dict) and fb.get("vetoed_reason"):
            st.warning(f"Agent's raw feedback was vetoed: {fb.get('vetoed_reason')}")
        if isinstance(fb, dict) and fb.get("rule_addenda"):
            st.caption("Rule addenda: " + "; ".join(fb["rule_addenda"]))
        triggers = cycle.get("revision_triggers") or []
        if triggers:
            st.markdown("**Why revision was triggered**")
            for t in triggers:
                st.write(f"- {t}")
    else:
        note = (cycle.get("stages") or {}).get("feedback", {}).get("note")
        st.info(note or "No feedback was sent (revision skipped).")

    # ---- STAGE 6 ----
    st.markdown(
        '<div class="stage-header">Stage 6 — Target LLM: revised response</div>',
        unsafe_allow_html=True,
    )
    if cycle.get("revised_response"):
        c1, c2 = st.columns(2)
        with c1:
            _response_card("Original", cycle.get("initial_response"))
        with c2:
            beh1 = cycle.get("revised_behavior") or {}
            _response_card(
                "Revised (from target model)",
                cycle.get("revised_response"),
                f"behavioral success={beh1.get('success')} · "
                f"flags={', '.join(beh1.get('flags') or beh1.get('bad_flags') or []) or 'none'}",
            )
        if cycle.get("revision_prompt"):
            with st.expander("Exact revision prompt sent to the target", expanded=False):
                st.code(cycle["revision_prompt"])
    else:
        note = (cycle.get("stages") or {}).get("revision", {}).get("note")
        st.info(note or "No revised response was generated.")

    # ---- STAGE 7 ----
    st.markdown(
        '<div class="stage-header">Stage 7 — Agent verification</div>',
        unsafe_allow_html=True,
    )
    ver = cycle.get("verification") or {}
    vstatus = ver.get("status") or "not_run"
    st.markdown(
        f'<div class="panel"><span class="badge {_status_class(vstatus)}">'
        f'Outcome · {_esc(vstatus)}</span>'
        f'&nbsp; <span class="badge">genuinely_better={_esc(ver.get("genuinely_better"))}</span>'
        f'<p class="soft" style="margin-top:0.8rem;color:#3a3a3c !important;">'
        f'{_esc(ver.get("reason") or ver.get("note") or "No verification reason recorded.")}</p></div>',
        unsafe_allow_html=True,
    )
    if ver.get("caveats"):
        st.caption("Caveats: " + "; ".join(ver["caveats"]))
    if ver.get("score_delta") is not None:
        st.caption(
            f"Judge score delta (secondary): {ver.get('score_delta')} · "
            f"comparable={ver.get('scores_comparable')}"
        )
    j1 = cycle.get("verify_judgment")
    if j1:
        # No nested expanders — Streamlit forbids expander-in-expander.
        st.markdown("**Post-feedback investigation trace**")
        render_timeline(j1, "Verification investigation", allow_expanders=False)

    # ---- STAGE 8 ----
    st.markdown(
        '<div class="stage-header">Stage 8 — Preference / training implication</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="panel">
          <p><strong>Three separate processes — do not conflate them:</strong></p>
          <ol>
            <li><strong>This cycle</strong> — in-session feedback and revision. Weights unchanged.</li>
            <li><strong>Preference storage</strong> — optional trusted pair for later training.</li>
            <li><strong>DPO / LoRA</strong> — separate Training tab job; only then can behavior permanently change.</li>
          </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )
    pref = cycle.get("preference")
    decision = cycle.get("preference_decision") or {}
    if cycle.get("saved_preference_id"):
        st.success(
            f"Trusted preference saved · id=`{cycle['saved_preference_id']}` · "
            "rejected=original · chosen=revised"
        )
    elif pref and (pref.get("eligibility") or {}).get("eligible") is False:
        st.warning(
            "Pair considered but **not eligible**: "
            + str((pref.get("eligibility") or {}).get("reason") or cycle.get("save_blocked_reason"))
        )
    elif cycle.get("save_blocked_reason"):
        st.warning(f"Preference not saved: {cycle.get('save_blocked_reason')}")
    elif decision.get("reason"):
        st.info(decision.get("reason"))
    else:
        st.info("No preference pair was created for this cycle.")
    if pref and pref.get("pair"):
        with st.expander("Preference pair preview", expanded=False):
            pair = pref["pair"]
            st.markdown("**Rejected (original)**")
            st.write(pair.get("rejected"))
            st.markdown("**Chosen (revised)**")
            st.write(pair.get("chosen"))
