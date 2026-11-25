"""AgenticRLHF-RLAIF V0.5 — Apple-inspired Streamlit demonstration."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.demo.agentic_panel import load_agentic_scenarios, render_cycle
from src.demo.prepare import DemoController, load_demo_scenarios, reset_demo_store
from src.eval.benchmark import (
    compare_runs,
    load_eval_set,
    load_preference_seeds,
    run_model_on_benchmark,
)
from src.models.ollama_client import OllamaClient
from src.models.target_llm import TransformersTarget
from src.pipeline.agentic_feedback import AgenticFeedbackLoop
from src.pipeline.cycle_schema import normalize_legacy_judgment
from src.pipeline.loop import DemoPipeline
from src.retrieval.chroma_store import get_safety_store
from src.training.dpo_train import find_latest_adapter, run_dpo_training
from src.training.feasibility import assess_training_feasibility

st.set_page_config(
    page_title="AgenticRLHF",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg: #f5f5f7;
  --surface: #ffffff;
  --ink: #1d1d1f;
  --muted: #3a3a3c;
  --line: rgba(0,0,0,0.10);
  --accent: #0071e3;
  --shadow: 0 14px 40px rgba(0,0,0,0.07);
  --radius: 20px;
}

/* Force readable light surfaces even when Streamlit/OS prefers dark */
html, body, .stApp, [data-testid="stAppViewContainer"], .block-container {
  color: var(--ink) !important;
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}
.stApp {
  background:
    radial-gradient(1100px 480px at 50% -8%, rgba(0,113,227,0.07), transparent 55%),
    linear-gradient(180deg, #ffffff 0%, #f5f5f7 45%, #eceef2 100%) !important;
}
#MainMenu, footer, header {visibility: hidden;}
.block-container { padding-top: 1.4rem; padding-bottom: 3.5rem; max-width: 1180px; }

/* Streamlit widgets: kill white-on-white */
label, [data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li, [data-testid="stCaption"] {
  color: var(--ink) !important;
}
.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
  color: #1d1d1f !important;
  background: #ffffff !important;
  border-color: rgba(0,0,0,0.14) !important;
}
div[data-testid="stExpander"] details {
  background: #ffffff !important;
  border: 1px solid var(--line) !important;
  border-radius: 14px !important;
}
div[data-testid="stExpander"] summary, div[data-testid="stExpander"] p {
  color: #1d1d1f !important;
}
div[data-testid="stTabs"] button { color: #1d1d1f !important; font-weight: 600 !important; }
div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] { color: #1d1d1f !important; }

.hero { text-align: center; padding: 2rem 1rem 1.2rem; }
.hero .eyebrow {
  color: var(--accent) !important; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; font-size: 0.78rem; margin-bottom: 0.7rem;
}
.hero h1 {
  font-size: clamp(2.2rem, 4.6vw, 3.6rem); line-height: 1.05; letter-spacing: -0.04em;
  font-weight: 700; margin: 0; color: #1d1d1f !important;
}
.hero p {
  margin: 0.85rem auto 0; max-width: 36rem; color: #3a3a3c !important;
  font-size: 1.08rem; line-height: 1.45;
}
.pipeline { display:flex; flex-wrap:wrap; gap:0.5rem; justify-content:center; margin:1.4rem 0 0.3rem; }
.pipeline span {
  background:#fff; border:1px solid var(--line); padding:0.42rem 0.85rem; border-radius:999px;
  font-size:0.84rem; font-weight:500; color:#1d1d1f !important;
}
.pipeline .arrow { border:none; background:transparent; color:#6e6e73 !important; padding:0.42rem 0.12rem; }

.status-row {
  display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:0.7rem;
  margin:1.2rem 0 1.5rem;
}
@media (max-width: 900px) { .status-row { grid-template-columns:repeat(2,minmax(0,1fr)); } }
.status-card {
  background:#fff; border:1px solid var(--line); border-radius:16px;
  padding:0.95rem 1rem; box-shadow:var(--shadow); min-height:88px;
}
.status-card .label {
  color:#6e6e73 !important; font-size:0.72rem; letter-spacing:0.04em;
  text-transform:uppercase; margin-bottom:0.3rem;
}
.status-card .value {
  font-size:0.96rem; font-weight:600; color:#1d1d1f !important; word-break:break-word;
}

.section-title {
  font-size:1.55rem; letter-spacing:-0.03em; font-weight:700;
  margin:0.15rem 0 0.3rem; color:#1d1d1f !important;
}
.section-sub { color:#3a3a3c !important; margin-bottom:1rem; line-height:1.45; }

.panel {
  background:#ffffff !important; border:1px solid var(--line); border-radius:var(--radius);
  padding:1.2rem 1.3rem; box-shadow:var(--shadow); margin-bottom:0.95rem; color:#1d1d1f !important;
}
.panel h3, .panel p, .panel li, .panel strong { color:#1d1d1f !important; }
.panel h3 { margin:0 0 0.7rem; font-size:1.02rem; letter-spacing:-0.02em; }
.soft { color:#3a3a3c !important; font-size:0.92rem; line-height:1.5; }

.badge {
  display:inline-flex; align-items:center; gap:0.3rem; border-radius:999px;
  padding:0.28rem 0.7rem; font-size:0.76rem; font-weight:600;
  border:1px solid var(--line); background:#f5f5f7; color:#1d1d1f !important;
}
.badge.llm { color:#0b6e4f !important; background:#eaf8f0; border-color:#cfeedd; }
.badge.hybrid { color:#7a5200 !important; background:#fff7e8; border-color:#ffe2a8; }
.badge.fallback { color:#9b1c1c !important; background:#feeceb; border-color:#f9c6c1; }
.badge.ok { color:#0b6e4f !important; background:#eaf8f0; }
.badge.warn { color:#7a5200 !important; background:#fff7e8; }

.resp {
  white-space:pre-wrap; line-height:1.55; font-size:1.02rem; color:#1d1d1f !important;
}
.score-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:0.65rem; }
.score-item {
  background:#f8f8fa; border:1px solid var(--line); border-radius:12px; padding:0.7rem 0.8rem;
}
.score-item .k { color:#6e6e73 !important; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.04em; }
.score-item .v { font-size:1.2rem; font-weight:700; color:#1d1d1f !important; margin-top:0.15rem; }

.stage-header {
  margin:1.35rem 0 0.55rem; font-size:1.12rem; font-weight:700;
  letter-spacing:-0.02em; color:#1d1d1f !important;
  border-left:4px solid var(--accent); padding-left:0.7rem;
}
.stage-rail { display:flex; flex-wrap:wrap; gap:0.45rem; margin:0.4rem 0 1rem; }
.stage-chip {
  display:inline-flex; align-items:center; gap:0.4rem; background:#fff;
  border:1px solid var(--line); border-radius:999px; padding:0.35rem 0.75rem;
  font-size:0.78rem; font-weight:600; color:#1d1d1f !important;
}
.stage-chip .stage-state { color:#6e6e73 !important; font-weight:500; }
.stage-dot {
  width:8px; height:8px; border-radius:50%; background:#c7c7cc; display:inline-block;
}
.stage-chip.ok .stage-dot { background:#34c759; }
.stage-chip.warn .stage-dot { background:#ff9f0a; }
.stage-chip.fallback .stage-dot, .stage-chip.hybrid .stage-dot { background:#ff3b30; }
.stage-chip.llm .stage-dot { background:#0071e3; }

.summary-row { display:flex; flex-wrap:wrap; gap:0.4rem; }
.feedback-panel { border-left:5px solid #0071e3; }

.trace-card {
  display:flex; gap:0.75rem; background:#fff; border:1px solid var(--line);
  border-radius:14px; padding:0.85rem 0.95rem; margin-bottom:0.55rem;
}
.trace-n {
  flex:0 0 1.6rem; height:1.6rem; border-radius:50%; background:#1d1d1f; color:#fff !important;
  display:flex; align-items:center; justify-content:center; font-size:0.75rem; font-weight:700;
}
.trace-title { font-weight:650; color:#1d1d1f !important; margin-bottom:0.2rem; }
.trace-text { color:#3a3a3c !important; font-size:0.92rem; line-height:1.45; }
.trace-decision { border-left:4px solid #0071e3; }
.trace-tool { border-left:4px solid #34c759; }
.trace-finish { border-left:4px solid #5856d6; }
.trace-note { border-left:4px solid #8e8e93; }
.trace-warn { border-left-color:#ff9f0a; }
.trace-error { border-left-color:#ff3b30; background:#fff8f7; }
</style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_pipeline() -> DemoPipeline:
    return DemoPipeline(session_id="streamlit")


def init_state() -> None:
    defaults = {
        "last_prompt": "",
        "last_response": "",
        "last_judgment": None,
        "last_mode": "playground",
        "train_log": "",
        "train_result": None,
        "before_results": None,
        "after_results": None,
        "comparison": None,
        "demo_last_run": None,
        "demo_comparison": None,
        "demo_train_result": None,
        "agentic_cycle": None,
        "agentic_scenario_meta": None,
        "agentic_custom_prompt": "",
        "agentic_eval_focus": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


@st.cache_resource
def get_demo_controller() -> DemoController:
    return DemoController()


def score_grid(scores: dict | None) -> None:
    items = list((scores or {}).items())
    if not items:
        st.caption("Scores: not produced for this judgment.")
        return
    html = ['<div class="score-grid">']
    for k, v in items:
        html.append(
            f'<div class="score-item"><div class="k">{k.replace("_"," ")}</div>'
            f'<div class="v">{v}<span style="color:#6e6e73;font-size:0.85rem">/5</span></div></div>'
        )
    html.append("</div>")
    st.markdown("\n".join(html), unsafe_allow_html=True)


def _safe_text(value: Any, fallback: str = "not produced") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def judgment_badge(j: dict) -> str:
    mode = j.get("judgment_mode") or ("heuristic_fallback" if j.get("used_fallback") else "llm")
    label = {
        "llm": "LLM JUDGMENT",
        "hybrid": "HYBRID (LLM + SCAFFOLD)",
        "heuristic_fallback": "HEURISTIC FALLBACK — NOT LLM JUDGMENT",
    }.get(mode, mode)
    css = {"llm": "llm", "hybrid": "hybrid", "heuristic_fallback": "fallback"}.get(mode, "")
    return f'<span class="badge {css}">{label}</span>'


def main() -> None:
    inject_css()
    init_state()
    pipe = get_pipeline()
    memory = pipe.memory
    store = get_safety_store()
    ollama = OllamaClient()
    adapter = find_latest_adapter()
    feas = assess_training_feasibility(n_preferences=memory.preference_count())
    retrieval = store.status

    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">AgenticRLHF · V{config.APP_VERSION}</div>
          <h1>AgenticRLHF</h1>
          <p>A supervisory agent investigates, retrieves context, and corrects the target in-session.</p>
          <div class="pipeline">
            <span>Generate</span><span class="arrow">→</span>
            <span>Agent investigates</span><span class="arrow">→</span>
            <span>Feedback</span><span class="arrow">→</span>
            <span>Revise</span><span class="arrow">→</span>
            <span>Verify</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    train_status = "Ready" if feas.trainable else "Blocked"
    if st.session_state.train_result:
        train_status = str(st.session_state.train_result.get("status", train_status)).title()
    elif adapter:
        train_status = "Adapter available"

    st.markdown(
        f"""
        <div class="status-row">
          <div class="status-card"><div class="label">Target</div><div class="value">{config.TARGET_OLLAMA_MODEL}</div></div>
          <div class="status-card"><div class="label">Judge</div><div class="value">{config.JUDGE_MODEL}</div></div>
          <div class="status-card"><div class="label">Retrieval</div><div class="value">{retrieval.get('backend')}</div></div>
          <div class="status-card"><div class="label">Preferences</div><div class="value">{memory.preference_count()}</div></div>
          <div class="status-card"><div class="label">Training</div><div class="value">{train_status}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(
        [
            "Agentic Loop",
            "Demo Mode",
            "Playground",
            "Judge",
            "Preference Data",
            "Training",
            "Evaluation",
        ]
    )
    demo = get_demo_controller()

    # ---------------- Agentic Loop (V0.5) ----------------
    with tabs[0]:
        st.markdown(
            '<div class="section-title">Agentic Feedback Loop</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-sub">Target answers → supervisory agent chooses tools, retrieves principles, '
            "sends actionable feedback → target revises → agent re-checks. "
            "Weights are not updated in this cycle; eligible pairs may be saved for later DPO.</div>",
            unsafe_allow_html=True,
        )
        agentic_scenarios = load_agentic_scenarios()
        labels = [f"{s['title']}" for s in agentic_scenarios] + ["Custom Prompt"]
        choice = st.selectbox("Scenario", labels, key="agentic_scenario_choice")
        evaluation_focus = ""
        if choice == "Custom Prompt":
            # Initialize widget keys once — do not pass `value=` with `key=` (Streamlit
            # then ignores edits until the next full remount, which broke demos).
            if "agentic_custom_prompt_box" not in st.session_state:
                st.session_state.agentic_custom_prompt_box = (
                    st.session_state.get("agentic_custom_prompt")
                    or st.session_state.get("last_prompt")
                    or ""
                )
            if "agentic_eval_focus_box" not in st.session_state:
                st.session_state.agentic_eval_focus_box = (
                    st.session_state.get("agentic_eval_focus") or ""
                )
            prompt = st.text_area(
                "Custom prompt (sent to the target model)",
                height=120,
                key="agentic_custom_prompt_box",
                placeholder="Type any prompt for a live demonstration…",
            )
            evaluation_focus = st.text_input(
                "Behavioral principle / evaluation focus (optional)",
                key="agentic_eval_focus_box",
                placeholder="e.g. refuse fabricating exact numbers; acknowledge uncertainty",
            )
            category = "custom"
            scenario_meta = {
                "id": "custom",
                "title": "Custom Prompt",
                "category": "custom",
                "why_demo": "Operator-supplied live prompt for demonstration.",
                "success_after_feedback": evaluation_focus.strip()
                or "Meaningful improvement against the stated focus, if any.",
            }
        else:
            scenario_meta = agentic_scenarios[labels.index(choice)]
            prompt = scenario_meta["prompt"]
            category = scenario_meta.get("category")
            evaluation_focus = scenario_meta.get("success_after_feedback") or ""
            st.markdown(
                f'<div class="panel"><strong>Why this demo:</strong> '
                f'{scenario_meta.get("why_demo")}<br/>'
                f'<span class="soft">Success after feedback: '
                f'{scenario_meta.get("success_after_feedback")}</span></div>',
                unsafe_allow_html=True,
            )

        # Exact prompt that will be sent to the target — preview only, same `prompt` as run().
        _preview = (prompt or "").strip()
        st.markdown(
            f"""
            <div class="panel" style="border-left:4px solid #0071e3;margin-top:0.35rem;">
              <h3 style="margin:0 0 0.55rem;color:#1d1d1f !important;">Prompt sent to Target LLM</h3>
              <div class="resp" style="color:#1d1d1f !important;background:#f8f8fa;border:1px solid rgba(0,0,0,0.10);border-radius:12px;padding:0.85rem 1rem;min-height:3.2rem;">
                {html.escape(_preview) if _preview else '<span class="soft" style="color:#6e6e73 !important;">(empty — enter a custom prompt above)</span>'}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        save_pref = st.checkbox(
            "Save preference if eligible (trusted gate)",
            value=True,
        )
        if st.button("Run Agentic Cycle", type="primary", use_container_width=True):
            if not (prompt or "").strip():
                st.error("Enter a prompt.")
            else:
                with st.spinner(
                    "Running target → agent tools → feedback → revise → verify…"
                ):
                    loop = AgenticFeedbackLoop(session_id="streamlit_agentic")
                    cycle = loop.run(
                        prompt.strip(),
                        category=category,
                        save_preference_if_eligible=save_pref,
                        evaluation_focus=evaluation_focus.strip() or None,
                        scenario=scenario_meta,
                    )
                    st.session_state.agentic_cycle = cycle
                    st.session_state.agentic_scenario_meta = scenario_meta
                    st.session_state.agentic_custom_prompt = prompt.strip()
                    st.session_state.agentic_eval_focus = evaluation_focus.strip()
                    st.session_state.last_prompt = prompt.strip()
                    st.session_state.last_response = cycle.get(
                        "revised_response"
                    ) or cycle.get("initial_response")
                    # Bridge agentic judgment → Judge tab without inventing fields
                    j0 = cycle.get("initial_judgment") or {}
                    j1 = cycle.get("verify_judgment") or {}
                    src = j1 if j1.get("scores") or j1.get("overall_score") is not None else j0
                    st.session_state.last_judgment = {
                        "result": {
                            "scores": src.get("scores") or {},
                            "overall_score": src.get("overall_score"),
                            "overall_assessment": src.get("overall_assessment")
                            or j0.get("overall_assessment"),
                            "critique": j0.get("critique"),
                            "score_explanation": j0.get("critique")
                            or src.get("overall_assessment"),
                            "confidence": src.get("confidence"),
                            "violations": j0.get("issues_found") or [],
                            "preferred_response": cycle.get("revised_response") or "",
                            "rejected_response": cycle.get("initial_response") or "",
                            "target_response": cycle.get("initial_response"),
                        },
                        "judgment_mode": j0.get("judgment_mode") or "agentic",
                        "used_fallback": j0.get("autonomy_mode")
                        == "fixed_workflow_fallback",
                        "judge_model": cycle.get("judge_model") or j0.get("judge_model"),
                        "retrieval_status": j0.get("retrieval_status") or {},
                        "retrieved_count": len(j0.get("retrieved_principles") or []),
                        "retrieved_principles": j0.get("retrieved_principles") or [],
                        "influential_principles": j0.get("influential_principles") or [],
                        "tool_calls": j0.get("tool_calls") or [],
                        "preference_eligibility": (
                            (cycle.get("preference") or {}).get("eligibility") or {}
                        ),
                        "verification": {
                            "ok": (cycle.get("verification") or {}).get("status")
                            in ("corrected", "improved", "no_revision_needed"),
                            "issues": [
                                (cycle.get("verification") or {}).get("reason")
                                or (cycle.get("verification") or {}).get("status")
                                or "see Agentic Loop tab"
                            ],
                        },
                    }

        if st.session_state.agentic_cycle:
            render_cycle(
                st.session_state.agentic_cycle,
                scenario_meta=st.session_state.get("agentic_scenario_meta"),
            )

    # ---------------- Demo Mode ----------------
    with tabs[1]:
        st.markdown('<div class="section-title">Demo Mode</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Reproducible faculty workflow: Baseline → Judge → Principles → Critique → Preference → DPO → Optimized → Compare. Uses an isolated demo database.</div>',
            unsafe_allow_html=True,
        )
        scenarios = load_demo_scenarios()
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            f"**Demo preferences:** {demo.preference_count()} &nbsp;·&nbsp; "
            f"**Main project preferences (unchanged by demo reset):** {memory.preference_count()}"
        )
        cprep1, cprep2, cprep3 = st.columns(3)
        if cprep1.button("Reset demo dataset only", use_container_width=True):
            info = reset_demo_store(backup=True)
            get_demo_controller.clear()
            demo = get_demo_controller()
            st.session_state.demo_last_run = None
            st.session_state.demo_comparison = None
            st.success(f"Demo DB reset. Backup: {info.get('backup')}")
        if cprep2.button("Run all train scenarios", type="primary", use_container_width=True):
            with st.spinner("Running demo scenarios (generate + judge + save if eligible)..."):
                runs = demo.run_all_train_scenarios(mode="playground")
            st.session_state.demo_last_run = runs[-1] if runs else None
            if st.session_state.demo_last_run:
                st.session_state.last_judgment = st.session_state.demo_last_run["judgment"]
                st.session_state.last_response = st.session_state.demo_last_run["generation"][
                    "response"
                ]
                st.session_state.last_prompt = st.session_state.demo_last_run["scenario"]["prompt"]
            st.success(f"Completed {len(runs)} scenarios. Demo prefs={demo.preference_count()}")
        if cprep3.button("Prepare note", use_container_width=True):
            st.info(
                "Reset clears only data/demo_rlaif.db. Main DB, checkpoints, and Chroma are preserved."
            )
        st.markdown("</div>", unsafe_allow_html=True)

        labels = [f"{s['id']} · {s['title']}" for s in scenarios]
        pick = st.selectbox("Scenario", labels)
        scenario = scenarios[labels.index(pick)]
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(f"**Category:** {scenario['category']}")
        st.markdown(f"**Expected principle:** {scenario['expected_principle']}")
        st.markdown("**Success looks like:** " + scenario["success_looks_like"])
        st.markdown("**Evaluation criteria**")
        for c in scenario["evaluation_criteria"]:
            st.write(f"- {c}")
        st.markdown(f"**Prompt**\n\n{scenario['prompt']}")
        st.markdown("</div>", unsafe_allow_html=True)

        d1, d2, d3 = st.columns(3)
        run_one = d1.button("1. Generate + Judge", type="primary", use_container_width=True)
        save_one = d2.button("2. Save preference (if eligible)", use_container_width=True)
        train_demo = d3.button("3. Run short DPO", use_container_width=True)

        if run_one:
            with st.spinner("Demo scenario running..."):
                out = demo.run_scenario(scenario["id"], mode="playground", save_if_eligible=False)
            st.session_state.demo_last_run = out
            st.session_state.last_judgment = out["judgment"]
            st.session_state.last_response = out["generation"]["response"]
            st.session_state.last_prompt = scenario["prompt"]

        out = st.session_state.demo_last_run
        if out and out.get("scenario", {}).get("id") == scenario["id"]:
            j = out["judgment"]
            res = j["result"]
            st.markdown('<div class="panel"><h3>Target response</h3>', unsafe_allow_html=True)
            st.markdown(f'<div class="resp">{out["generation"]["response"]}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown('<div class="panel"><h3>Judge verdict</h3>', unsafe_allow_html=True)
            st.markdown(judgment_badge(j), unsafe_allow_html=True)
            score_grid(res["scores"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Overall", res["overall_score"])
            m2.metric("Confidence", f"{res['confidence']:.2f}")
            m3.metric("Retrieved", j.get("retrieved_count", 0))
            st.markdown("**Critique**")
            st.write(res["critique"])
            elig = j.get("preference_eligibility") or {}
            if elig.get("eligible"):
                st.markdown('<span class="badge ok">Preference eligible to save</span>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<span class="badge fallback">Not eligible: {elig.get("reason")}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

            if save_one:
                if not elig.get("eligible"):
                    st.error(f"Blocked: {elig.get('reason')}")
                else:
                    try:
                        pid = demo.pipeline.save_preference_pair(j["preference_pair"])
                        st.success(f"Saved demo preference {pid}")
                    except Exception as e:
                        st.error(str(e))

        if train_demo:
            with st.spinner("DPO on demo preferences..."):
                log_box = st.empty()
                lines: list[str] = []

                def _log(msg: str) -> None:
                    lines.append(msg)
                    log_box.code("\n".join(lines[-60:]))

                result = demo.train_demo(max_steps=4, log=_log)
            st.session_state.demo_train_result = result
            if result.get("status") == "completed" and result.get("adapter_verified"):
                st.success(f"DPO completed. Adapter verified: {result.get('checkpoint_path')}")
                pipe.reload_optimized()
                get_pipeline.clear()
            else:
                st.error(
                    f"Training status={result.get('status')} · {result.get('error') or 'see log'}"
                )

        st.markdown('<div class="panel"><h3>4. Before / After on demo prompts</h3>', unsafe_allow_html=True)
        if st.button("Compare HF baseline vs HF+adapter on demo scenarios", use_container_width=True):
            with st.spinner("Running honest before/after (adapter load verified)..."):
                st.session_state.demo_comparison = demo.compare_demo_scenarios()
        cmp = st.session_state.demo_comparison
        if cmp:
            st.caption(cmp.get("benchmark_note"))
            if not cmp.get("optimized_ready"):
                st.warning(
                    "Optimized path not ready — adapter missing or failed to load. "
                    "Baseline results only are shown."
                )
            else:
                st.markdown(
                    '<span class="badge ok">Optimized inference uses loaded LoRA adapter</span>',
                    unsafe_allow_html=True,
                )
            if cmp.get("comparison"):
                c = cmp["comparison"]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Avg before", c["avg_before"])
                m2.metric("Avg after", c["avg_after"], delta=c["avg_delta"])
                m3.metric("Improved", c["improved"])
                m4.metric("Regressed", c["regressed"])
                for row in c["rows"]:
                    with st.expander(
                        f"{row['id']} · {row['status']} · Δ {row['delta']} "
                        f"({row['before_score']} → {row['after_score']})"
                    ):
                        st.write(row["prompt"])
                        a, b = st.columns(2)
                        with a:
                            st.markdown("**Baseline**")
                            st.write(row["before_response"])
                            st.json(row["before_scores"])
                        with b:
                            st.markdown("**Optimized**")
                            st.write(row["after_response"])
                            st.json(row["after_scores"])
            elif cmp.get("before"):
                st.info("Baseline responses collected; optimized comparison unavailable.")
                for row in cmp["before"]:
                    with st.expander(f"{row['id']} · score={row['overall_score']}"):
                        st.write(row["response"])
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Playground ----------------
    with tabs[2]:
        st.markdown('<div class="section-title">Playground</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Generate a response from the target model, then evaluate it with the supervisory judge.</div>',
            unsafe_allow_html=True,
        )

        seeds = load_preference_seeds()
        seed_labels = [f"{s['id']} · {s['category']}" for s in seeds]
        available = ollama.list_models()
        qwen_opts = [m for m in available if "qwen" in m.lower()] or [config.TARGET_OLLAMA_MODEL]
        mode_options = {
            "Playground (Ollama)": "playground",
            "Trainable base (HF SmolLM2)": "train_base",
            "Optimized (HF + LoRA adapter)": "optimized",
        }

        c1, c2 = st.columns([1.15, 0.85])
        with c1:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            mode_label = st.selectbox("Target path", list(mode_options.keys()))
            mode = mode_options[mode_label]
            ollama_model = config.TARGET_OLLAMA_MODEL
            if mode == "playground":
                ollama_model = st.selectbox("Ollama target model", qwen_opts, index=0)
            if mode == "optimized" and adapter is None:
                st.warning("No adapter checkpoint found. Train first, or use another path.")
            pick = st.selectbox("Seed prompt", ["(custom)"] + seed_labels)
            category = None
            default_prompt = st.session_state.last_prompt
            if pick != "(custom)":
                item = seeds[seed_labels.index(pick)]
                default_prompt = item["prompt"]
                category = item.get("category")
            prompt = st.text_area("Prompt", value=default_prompt, height=130)
            b1, b2, b3 = st.columns(3)
            gen_clicked = b1.button("Generate", use_container_width=True)
            both_clicked = b2.button("Generate + Judge", type="primary", use_container_width=True)
            judge_clicked = b3.button("Judge only", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if (gen_clicked or both_clicked) and prompt.strip():
                with st.spinner("Generating..."):
                    try:
                        out = pipe.generate(prompt, mode=mode, ollama_model=ollama_model)
                        st.session_state.last_prompt = prompt
                        st.session_state.last_response = out["response"]
                        st.session_state.last_mode = mode
                        st.session_state.last_judgment = None
                    except Exception as e:
                        st.error(str(e))

            if (judge_clicked or both_clicked) and st.session_state.last_response:
                with st.spinner("Judge evaluating..."):
                    judgment = pipe.judge_response(
                        st.session_state.last_prompt or prompt,
                        st.session_state.last_response,
                        category=category,
                    )
                    st.session_state.last_judgment = judgment

            if st.session_state.last_response:
                st.markdown('<div class="panel"><h3>Target response</h3>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="resp">{st.session_state.last_response}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(f"Mode: {st.session_state.last_mode}")
                st.markdown("</div>", unsafe_allow_html=True)

            j = st.session_state.last_judgment
            if j:
                res = j["result"]
                st.markdown('<div class="panel"><h3>Judge snapshot</h3>', unsafe_allow_html=True)
                st.markdown(judgment_badge(j), unsafe_allow_html=True)
                st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)
                score_grid(res["scores"])
                m1, m2, m3 = st.columns(3)
                m1.metric("Overall", res["overall_score"])
                m2.metric("Confidence", f"{res['confidence']:.2f}")
                m3.metric("Retrieved", j.get("retrieved_count", 0))
                st.markdown("**Critique**")
                st.write(res["critique"])
                st.markdown("**Preferred response**")
                st.write(res["preferred_response"])
                elig = j.get("preference_eligibility") or {}
                if elig.get("eligible"):
                    st.markdown('<span class="badge ok">Eligible to save</span>', unsafe_allow_html=True)
                    if st.button("Save preference pair", type="primary"):
                        try:
                            pid = pipe.save_preference_pair(j["preference_pair"])
                            st.success(f"Saved {pid}")
                        except Exception as e:
                            st.error(str(e))
                else:
                    st.markdown(
                        f'<span class="badge fallback">Not saved silently — {elig.get("reason")}</span>',
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="panel"><h3>Path notes</h3>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="soft">
                <b>Playground</b> uses Ollama for higher-quality local responses.<br><br>
                <b>Optimized</b> uses Hugging Face SmolLM2 + the trained LoRA adapter.
                Ollama cannot load HF PEFT adapters.<br><br>
                The target never receives judge tools, retrieval, or agent orchestration.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.json(pipe.get_active_target().info())
            st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Judge ----------------
    with tabs[3]:
        st.markdown('<div class="section-title">Judge</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Supervisory agent evidence, tools, retrieval, and structured judgment. No chain-of-thought.</div>',
            unsafe_allow_html=True,
        )
        j_raw = st.session_state.last_judgment
        j = normalize_legacy_judgment(j_raw)
        if not j:
            st.info("Run a judgment in Demo Mode, Playground, or Agentic Loop first.")
        else:
            res = j["result"]
            st.markdown(judgment_badge(j), unsafe_allow_html=True)
            missing = j.get("missing_fields") or []
            if missing:
                st.caption(
                    "Optional fields not produced by this run: "
                    + ", ".join(missing[:12])
                    + ("…" if len(missing) > 12 else "")
                )
            st.caption(
                f"Judge model: {j.get('judge_model') or 'n/a'} · Retrieval: "
                f"{(j.get('retrieval_status') or {}).get('backend') or 'n/a'} · "
                f"Docs: {j.get('retrieved_count', 0)}"
            )
            st.markdown('<div class="panel"><h3>Target response</h3>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="resp">{_safe_text(res.get("target_response") or st.session_state.last_response)}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="panel"><h3>Verdict & scores</h3>', unsafe_allow_html=True)
            st.write(_safe_text(res.get("overall_assessment")))
            score_grid(res.get("scores"))
            m1, m2 = st.columns(2)
            overall = res.get("overall_score")
            conf = res.get("confidence")
            m1.metric("Overall", overall if overall is not None else "n/a")
            m2.metric(
                "Confidence",
                f"{conf:.2f}" if isinstance(conf, (int, float)) else "n/a",
            )
            st.markdown("</div>", unsafe_allow_html=True)

            left, right = st.columns(2)
            with left:
                st.markdown(
                    '<div class="panel"><h3>Influential retrieved principles</h3>',
                    unsafe_allow_html=True,
                )
                influ = j.get("influential_principles") or j.get("retrieved_principles") or []
                if not influ:
                    st.caption("No retrieved principles.")
                for p in influ:
                    with st.expander(
                        f"{p.get('title') or p.get('id')} · {p.get('category')} · score={p.get('score')}"
                    ):
                        st.write(p.get("excerpt") or p.get("text") or p.get("principle"))
                        st.caption(f"backend={p.get('backend')}")
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown('<div class="panel"><h3>Tools used</h3>', unsafe_allow_html=True)
                calls = j.get("tool_calls") or []
                if not calls:
                    st.caption("No tool calls recorded for this judgment.")
                for call in calls:
                    st.markdown(f"**{call.get('tool') or 'unknown'}**")
                    st.json(call.get("result_preview"))
                st.markdown("</div>", unsafe_allow_html=True)

            with right:
                st.markdown('<div class="panel"><h3>Critique</h3>', unsafe_allow_html=True)
                st.write(_safe_text(res.get("critique")))
                st.markdown("**Evidence / explanation**")
                st.write(_safe_text(res.get("score_explanation")))
                if res.get("violations"):
                    st.markdown("**Failure labels**")
                    for v in res["violations"]:
                        st.write(f"- {v}")
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown(
                    '<div class="panel"><h3>Preference pair from this evaluation</h3>',
                    unsafe_allow_html=True,
                )
                st.markdown("**Chosen**")
                st.success(_safe_text(res.get("preferred_response"), "not produced"))
                st.markdown("**Rejected**")
                st.error(_safe_text(res.get("rejected_response"), "not produced"))
                elig = j.get("preference_eligibility") or {}
                if elig.get("eligible"):
                    st.markdown(
                        '<span class="badge ok">Eligible to save</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<span class="badge fallback">Not eligible: '
                        f'{elig.get("reason") or "n/a"}</span>',
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

                ver = j.get("verification") or {"present": False, "ok": False, "issues": []}
                if not ver.get("present"):
                    st.caption("Schema validation metadata: not produced for this run.")
                elif ver.get("ok"):
                    st.markdown(
                        '<span class="badge ok">Validation passed</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.error(
                        "Validation issues: "
                        + "; ".join(ver.get("issues") or ["unknown"])
                    )

    # ---------------- Preference Data ----------------
    with tabs[4]:
        st.markdown('<div class="section-title">Preference Data</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Chosen / rejected pairs used for preference optimization.</div>',
            unsafe_allow_html=True,
        )
        q = st.text_input("Filter records", "")
        prefs = memory.list_preferences(limit=500, query=q or None)
        st.metric("Dataset size", memory.preference_count())
        if not prefs:
            st.info("No preference pairs saved yet.")
        for p in prefs:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown(
                f"**{p['id'][:8]}…** · score={p.get('overall_score')} · "
                f"confidence={p.get('confidence')} · {p.get('category') or 'n/a'} · {p.get('source')}"
            )
            st.markdown("**Prompt**")
            st.write(p["prompt"])
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Chosen**")
                st.success(p["chosen"])
            with c2:
                st.markdown("**Rejected**")
                st.error(p["rejected"])
            if p.get("rationale"):
                st.markdown("**Judge rationale**")
                st.write(p["rationale"])
            if p.get("retrieved_references"):
                st.caption(
                    "Retrieval refs: "
                    + ", ".join(
                        str(r.get("id")) for r in p["retrieved_references"] if r.get("id")
                    )
                )
            if p.get("scores"):
                st.json(p["scores"])
            st.markdown("</div>", unsafe_allow_html=True)

        export_path = config.EXPORTS_DIR / "preferences.jsonl"
        if st.button("Export JSONL"):
            n = memory.export_preferences_jsonl(export_path)
            st.success(f"Exported {n} pairs → {export_path}")

    # ---------------- Training ----------------
    with tabs[5]:
        st.markdown('<div class="section-title">Training</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Real DPO + LoRA on the Hugging Face SmolLM2 path. Not prompt rewriting.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            f"**Base model for training:** `{config.TARGET_HF_MODEL_ID}`  \n"
            f"**Method:** DPO + LoRA (TRL)  \n"
            f"**Playground Ollama model:** `{config.TARGET_OLLAMA_MODEL}` "
            "(not used for adapter training)"
        )
        if feas.trainable:
            st.markdown('<span class="badge ok">Trainable in this environment</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge fallback">Training blocked</span>', unsafe_allow_html=True)
            st.write("; ".join(feas.reasons))
        for w in feas.warnings:
            st.warning(w)
        st.json(feas.recommended_config)
        st.markdown("</div>", unsafe_allow_html=True)

        max_steps = st.slider("Max steps", 2, 40, int(config.TRAIN_MAX_STEPS))
        if st.button("Start DPO + LoRA", type="primary", disabled=not feas.trainable):
            log_box = st.empty()
            lines: list[str] = []

            def _log(msg: str) -> None:
                lines.append(msg)
                log_box.code("\n".join(lines[-80:]))

            with st.spinner("Training — waiting for real completion status..."):
                pipe.unload_all_targets()
                result = run_dpo_training(memory=memory, max_steps=max_steps, log=_log)
            st.session_state.train_result = result
            st.session_state.train_log = result.get("log_text", "")
            if result.get("status") == "completed":
                st.success(f"Completed. Adapter: {result.get('checkpoint_path')}")
                pipe.reload_optimized()
                get_pipeline.clear()
            else:
                st.error(
                    f"Status: {result.get('status')} — {result.get('error') or 'see logs'}"
                )

        if st.session_state.train_result:
            st.markdown('<div class="panel"><h3>Last run</h3>', unsafe_allow_html=True)
            st.json(
                {
                    k: v
                    for k, v in st.session_state.train_result.items()
                    if k not in {"log_text", "feasibility"}
                }
            )
            st.code(st.session_state.train_log or "(empty)")
            st.markdown("</div>", unsafe_allow_html=True)

        latest = find_latest_adapter()
        st.markdown('<div class="panel"><h3>Checkpoint</h3>', unsafe_allow_html=True)
        if latest:
            st.success(str(latest))
            meta = latest / "train_meta.json"
            if meta.exists():
                st.json(json.loads(meta.read_text(encoding="utf-8")))
        else:
            st.info("No adapter yet.")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Initialize / refresh Chroma retrieval corpus"):
            if st.button("Populate Chroma from safety files"):
                out = store.initialize_from_files(force=True)
                st.json(out)
            st.json(store.status)

    # ---------------- Evaluation ----------------
    with tabs[6]:
        st.markdown('<div class="section-title">Evaluation</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Small demonstration benchmark only — not a general safety claim. Same judge scores both sides (evaluator bias possible). Optimized path requires a successfully loaded LoRA adapter.</div>',
            unsafe_allow_html=True,
        )
        eval_items = load_eval_set()
        max_items = st.slider("Items", 2, len(eval_items), min(4, len(eval_items)))
        c1, c2 = st.columns(2)
        run_before = c1.button("Run baseline (HF SmolLM2)", use_container_width=True)
        adapter_path = find_latest_adapter()
        run_after = c2.button(
            "Run optimized (HF + LoRA)",
            use_container_width=True,
            disabled=adapter_path is None,
        )

        if run_before:
            with st.spinner("Evaluating baseline..."):
                base = TransformersTarget(adapter_path=None)
                st.session_state.before_results = run_model_on_benchmark(
                    base, items=eval_items, judge=pipe.judge, max_items=max_items
                )
                for row in st.session_state.before_results:
                    row["adapter_loaded"] = False
                if st.session_state.after_results:
                    st.session_state.comparison = compare_runs(
                        st.session_state.before_results, st.session_state.after_results
                    )

        if run_after:
            if not adapter_path:
                st.error("No adapter checkpoint found.")
            else:
                with st.spinner("Loading adapter and evaluating optimized model..."):
                    opt = TransformersTarget(adapter_path=adapter_path)
                    try:
                        opt.load()
                    except Exception as e:
                        st.error(f"Adapter failed to load — not labeling as optimized. {e}")
                        opt = None
                    if opt is not None and not opt.info().get("adapter_loaded"):
                        st.error("Adapter did not load; refusing optimized evaluation.")
                        opt = None
                    if opt is not None:
                        st.markdown(
                            '<span class="badge ok">Adapter loaded on matching base model</span>',
                            unsafe_allow_html=True,
                        )
                        st.json(
                            {
                                "model_id": opt.info().get("model_id"),
                                "adapter_path": opt.info().get("adapter_path"),
                                "adapter_loaded": True,
                                "adapter_base_model": opt.info().get("adapter_base_model"),
                            }
                        )
                        st.session_state.after_results = run_model_on_benchmark(
                            opt, items=eval_items, judge=pipe.judge, max_items=max_items
                        )
                        for row in st.session_state.after_results:
                            row["adapter_loaded"] = True
                        if st.session_state.before_results:
                            st.session_state.comparison = compare_runs(
                                st.session_state.before_results, st.session_state.after_results
                            )

        cmp = st.session_state.comparison
        if cmp:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg before", cmp["avg_before"])
            m2.metric("Avg after", cmp["avg_after"], delta=cmp["avg_delta"])
            m3.metric("Improved", cmp["improved"])
            m4.metric("Regressed", cmp["regressed"])
            for row in cmp["rows"]:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown(
                    f"**{row['id']}** · {row['status']} · Δ {row['delta']} "
                    f"({row['before_score']} → {row['after_score']})"
                )
                st.write(row["prompt"])
                a, b = st.columns(2)
                with a:
                    st.markdown("**Baseline response**")
                    st.write(row["before_response"])
                    st.markdown("**Baseline scores**")
                    st.json(row["before_scores"])
                with b:
                    st.markdown("**Optimized response**")
                    st.write(row["after_response"])
                    st.markdown("**Optimized scores**")
                    st.json(row["after_scores"])
                st.caption(
                    "Behavioral change is measured by the same judge rubric on identical prompts. "
                    "Improvements are not guaranteed."
                )
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Run baseline and optimized evaluations to compare measured scores.")


if __name__ == "__main__":
    main()
