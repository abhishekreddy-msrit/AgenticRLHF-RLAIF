"""E2E: run three faculty scenarios + custom prompt through AgenticFeedbackLoop."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.demo.agentic_panel import bridge_to_legacy_judgment, normalize_agentic_result, render_cycle
from src.pipeline.agentic_feedback import AgenticFeedbackLoop


def _assert_renderable(cycle: dict, label: str) -> dict:
    n = normalize_agentic_result(cycle)
    bridged = bridge_to_legacy_judgment(cycle)
    res = bridged["result"]
    for key in (
        "score_explanation",
        "critique",
        "preferred_response",
        "rejected_response",
        "scores",
        "overall_score",
        "confidence",
    ):
        assert key in res, f"{label}: missing bridge key {key}"
        _ = res[key]
    # Touch fields the staged UI reads
    _ = n["initial_judgment"]["action_trace"]
    _ = n["verification"]["status"]
    _ = n.get("feedback_to_target")
    return n


def main() -> int:
    scenarios = json.loads(
        (ROOT / "data" / "benchmark" / "agentic_demo_scenarios.json").read_text(encoding="utf-8")
    )["scenarios"]
    # Top 3 optimized demos
    want = scenarios[:3]
    loop = AgenticFeedbackLoop(session_id="e2e_agentic_ui")
    report = []

    for s in want:
        t0 = time.time()
        print(f"\n=== RUN {s['id']} ===", flush=True)
        cycle = loop.run(
            s["prompt"],
            category=s.get("category"),
            save_preference_if_eligible=False,
            scenario_title=s.get("title"),
            scenario_why=s.get("why_demo"),
        )
        n = _assert_renderable(cycle, s["id"])
        elapsed = round(time.time() - t0, 1)
        entry = {
            "id": s["id"],
            "ok": n.get("ok") and not n.get("error"),
            "error": n.get("error"),
            "autonomy": n["initial_judgment"].get("autonomy_mode"),
            "tools_used": sorted(
                {
                    (step.get("detail") or {}).get("tool")
                    for step in (n["initial_judgment"].get("action_trace") or [])
                    if isinstance(step, dict) and step.get("event") == "tool_selected"
                }
                - {None}
            ),
            "issues": n["initial_judgment"].get("issues_found"),
            "revised": bool(n.get("revised_response")),
            "verification": n["verification"].get("status"),
            "genuinely_better": n["verification"].get("genuinely_better"),
            "initial_head": (n.get("initial_response") or "")[:160],
            "revised_head": (n.get("revised_response") or "")[:160],
            "elapsed_s": elapsed,
        }
        report.append(entry)
        print(json.dumps(entry, indent=2, ensure_ascii=False), flush=True)

    # Custom prompt
    print("\n=== RUN custom ===", flush=True)
    t0 = time.time()
    custom = loop.run(
        "Name one benefit of brushing teeth twice a day in a single short sentence.",
        category="custom",
        save_preference_if_eligible=False,
        scenario_title="Custom prompt",
        evaluation_focus="Keep the answer to one short sentence; no lecture.",
    )
    n = _assert_renderable(custom, "custom")
    report.append(
        {
            "id": "custom",
            "ok": n.get("ok") and not n.get("error"),
            "error": n.get("error"),
            "autonomy": n["initial_judgment"].get("autonomy_mode"),
            "verification": n["verification"].get("status"),
            "revised": bool(n.get("revised_response")),
            "elapsed_s": round(time.time() - t0, 1),
            "initial_head": (n.get("initial_response") or "")[:160],
        }
    )

    # Sparse result must not break normalize (already covered) — simulate incomplete
    sparse_n = normalize_agentic_result({"prompt": "x", "initial_judgment": {"scores": {}}})
    assert sparse_n["initial_judgment"]["score_explanation"]

    out = ROOT / "exports" / "agentic_e2e_ui_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)
    failed = [r for r in report if not r.get("ok")]
    if failed:
        print("FAILED:", [f["id"] for f in failed], flush=True)
        return 1
    print("ALL E2E CYCLES OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
