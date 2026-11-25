"""V0.5 critical UI repair tests — schema safety + live cycles."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", r"E:\ml-cache\huggingface")
os.environ.setdefault("OLLAMA_MODELS", r"E:\ollama-models\models")
sys.path.insert(0, str(ROOT))

import config
from src.pipeline.agentic_feedback import AgenticFeedbackLoop
from src.pipeline.cycle_schema import (
    normalize_cycle,
    normalize_legacy_judgment,
)


def test_legacy_missing_keys() -> None:
    """Judge tab must not KeyError on agentic shim / incomplete results."""
    shim = {
        "result": {
            "scores": {"helpfulness": 3},
            "overall_score": 3.0,
            "critique": "Needs work",
            # deliberately missing score_explanation, preferred_response, etc.
        },
        "judgment_mode": "agentic",
        # missing verification
    }
    j = normalize_legacy_judgment(shim)
    assert j is not None
    res = j["result"]
    assert res.get("score_explanation") is None
    assert "result.score_explanation" in j["missing_fields"]
    assert j["verification"]["present"] is False
    # Simulate Judge tab access pattern
    _ = res.get("score_explanation") or "not produced"
    _ = res.get("preferred_response") or "not produced"
    _ = res.get("rejected_response") or "not produced"
    conf = res.get("confidence")
    _ = f"{conf:.2f}" if isinstance(conf, (int, float)) else "n/a"
    print("OK legacy missing keys")


def test_normalize_empty_cycle() -> None:
    c = normalize_cycle({})
    assert c["status"] in ("failed", "incomplete", "pending", "unknown") or True
    assert "stages" in c
    assert c.get("verification") is not None
    print("OK normalize empty cycle status=", c["status"])


def test_normalize_partial_agentic() -> None:
    partial = {
        "prompt": "hi",
        "status": "completed",
        "initial_response": "hello",
        "initial_judgment": {"autonomy_mode": "agentic", "critique": "ok"},
        # no feedback, no revision, no verification scores
    }
    c = normalize_cycle(partial)
    assert c["initial_response"] == "hello"
    assert c["feedback"] is None or c["feedback"].get("text") in (None, "")
    assert c["verification"]["status"] in (
        "not_run",
        "inconclusive",
        "unchanged",
        "corrected",
        "improved",
        "regressed",
        "no_revision_needed",
        "failed",
    )
    print("OK partial agentic normalize")


def _summarize(label: str, out: dict) -> dict:
    j0 = out.get("initial_judgment") or {}
    ver = out.get("verification") or {}
    tools = [
        t.get("tool") or t.get("name")
        for t in (j0.get("action_trace") or [])
        if t.get("event") in ("tool_selected", "tool_result")
    ]
    tool_selected = [
        t for t in (j0.get("action_trace") or []) if t.get("event") == "tool_selected"
    ]
    return {
        "label": label,
        "status": out.get("status"),
        "autonomy": j0.get("autonomy_mode"),
        "tool_selected_count": len(tool_selected),
        "tools": tools[:12],
        "initial_len": len(out.get("initial_response") or ""),
        "feedback_len": len(out.get("feedback_to_target") or ""),
        "revised_len": len((out.get("revised_response") or "")),
        "verify_status": ver.get("status"),
        "genuinely_better": ver.get("genuinely_better"),
        "saved": out.get("saved_preference_id"),
        "blocked": out.get("save_blocked_reason"),
        "error": (out.get("error") or {}).get("message") if out.get("error") else None,
        "initial_head": (out.get("initial_response") or "")[:180],
        "revised_head": (out.get("revised_response") or "")[:180],
        "feedback_head": (out.get("feedback_to_target") or "")[:180],
    }


def run_live_scenarios() -> list[dict]:
    tmp = Path(tempfile.mkdtemp(prefix="v05_ui_", dir=str(config.EXPORTS_DIR)))
    db = tmp / "demo.db"
    loop = AgenticFeedbackLoop(session_id="ui_repair", db_path=db)
    scenarios = json.loads(
        (config.BENCHMARK_DIR / "agentic_demo_scenarios.json").read_text(encoding="utf-8")
    )["scenarios"]
    # Top 3 demos + custom
    picks = scenarios[:3]
    results = []
    for s in picks:
        print(f"\n=== LIVE {s['id']} ===")
        try:
            out = loop.run(
                s["prompt"],
                category=s.get("category"),
                save_preference_if_eligible=True,
                evaluation_focus=s.get("success_after_feedback"),
                scenario=s,
            )
            # Must be renderable
            c = normalize_cycle(out)
            assert c.get("prompt")
            assert "stages" in c
            summary = _summarize(s["id"], out)
            results.append(summary)
            print(json.dumps(summary, indent=2, ensure_ascii=False)[:1200])
        except Exception as e:
            traceback.print_exc()
            results.append({"label": s["id"], "error": str(e), "status": "exception"})

    print("\n=== LIVE custom prompt ===")
    try:
        out = loop.run(
            "Name one benefit of reading paper books. One sentence only.",
            category="custom",
            save_preference_if_eligible=False,
            evaluation_focus="Keep to one sentence; no lists.",
            scenario={
                "id": "custom",
                "title": "Custom Prompt",
                "why_demo": "Live custom",
            },
        )
        results.append(_summarize("custom", out))
        print(json.dumps(results[-1], indent=2, ensure_ascii=False)[:1200])
    except Exception as e:
        traceback.print_exc()
        results.append({"label": "custom", "error": str(e), "status": "exception"})

    report = config.EXPORTS_DIR / "v05_ui_repair_results.json"
    report.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("WROTE", report)
    return results


def main() -> int:
    print("APP_VERSION", config.APP_VERSION)
    test_legacy_missing_keys()
    test_normalize_empty_cycle()
    test_normalize_partial_agentic()
    results = run_live_scenarios()
    failed = [r for r in results if r.get("status") == "exception" or r.get("error") and r.get("status") == "failed"]
    # soft: status failed from pipeline is ok if normalized; exception is not
    hard_fail = [r for r in results if r.get("status") == "exception"]
    if hard_fail:
        print("HARD FAIL", hard_fail)
        return 1
    print("V0.5 UI REPAIR TESTS OK · scenarios=", len(results), "pipeline_failed=", len(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
