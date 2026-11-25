"""Run the three faculty demo scenarios and print a compact evidence report."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", r"E:\ml-cache\huggingface")
os.environ.setdefault("OLLAMA_MODELS", r"E:\ollama-models\models")
sys.path.insert(0, str(ROOT))

import config
from src.pipeline.agentic_feedback import AgenticFeedbackLoop
from src.pipeline.cycle_schema import build_timeline, tool_usage


def main() -> int:
    scenarios = json.loads(
        (config.BENCHMARK_DIR / "agentic_demo_scenarios.json").read_text(encoding="utf-8")
    )["scenarios"][:3]
    tmp = Path(tempfile.mkdtemp(prefix="demo3_", dir=str(config.EXPORTS_DIR)))
    loop = AgenticFeedbackLoop(session_id="demo3", db_path=tmp / "d.db")
    report = []
    for s in scenarios:
        print("\n" + "=" * 60)
        print(s["id"], "·", s["title"])
        out = loop.run(
            s["prompt"],
            category=s.get("category"),
            save_preference_if_eligible=True,
            evaluation_focus=s.get("success_after_feedback"),
            scenario=s,
        )
        j0 = out.get("initial_judgment") or {}
        usage = tool_usage(j0)
        cards = build_timeline(j0)
        row = {
            "id": s["id"],
            "status": out.get("status"),
            "autonomy": j0.get("autonomy_mode"),
            "tools_used": usage.get("used"),
            "tools_not_used": usage.get("not_used"),
            "timeline_kinds": [c.get("kind") for c in cards],
            "timeline_titles": [c.get("title") for c in cards[:8]],
            "triggers": out.get("revision_triggers"),
            "beh_success": (out.get("initial_behavior") or {}).get("success"),
            "beh_flags": (out.get("initial_behavior") or {}).get("flags")
            or ((out.get("initial_behavior") or {}).get("behavior") or {}).get("flags"),
            "initial_head": (out.get("initial_response") or "")[:220],
            "feedback_head": (out.get("feedback_to_target") or "")[:220],
            "revised_head": (out.get("revised_response") or "")[:220],
            "verify": (out.get("verification") or {}).get("status"),
            "genuinely_better": (out.get("verification") or {}).get("genuinely_better"),
            "saved": out.get("saved_preference_id"),
            "blocked": out.get("save_blocked_reason"),
        }
        report.append(row)
        print(json.dumps(row, indent=2, ensure_ascii=False))
    path = config.EXPORTS_DIR / "demo_three_scenarios.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nWROTE", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
