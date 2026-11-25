"""Ensure agentic→Judge bridge never KeyErrors on optional fields."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.demo.agentic_panel import bridge_to_legacy_judgment, normalize_agentic_result


def main() -> int:
    sparse = {
        "prompt": "hi",
        "initial_response": "hello",
        "initial_judgment": {
            "autonomy_mode": "agentic",
            "scores": {"safety": 4},
            "issues_found": [],
            # deliberately missing score_explanation / critique
        },
        "verification": {"status": "no_revision_needed"},
    }
    n = normalize_agentic_result(sparse)
    assert n["initial_judgment"]["score_explanation"]
    bridged = bridge_to_legacy_judgment(sparse)
    res = bridged["result"]
    # These are the keys Judge tab used to crash on
    for key in (
        "score_explanation",
        "critique",
        "preferred_response",
        "rejected_response",
        "scores",
        "overall_score",
        "confidence",
    ):
        assert key in res, key
        _ = res[key]
    ver = bridged["verification"]
    assert "ok" in ver
    print("UI RENDER SAFETY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
