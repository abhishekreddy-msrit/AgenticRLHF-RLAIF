"""Faculty Demo Mode smoke test — isolated demo DB, no model upgrades."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.demo.prepare import DemoController, load_demo_scenarios, reset_demo_store
from src.models.target_llm import TransformersTarget
from src.training.dpo_train import find_latest_adapter


def main() -> int:
    print("=== reset demo db ===")
    print(reset_demo_store(backup=True))

    scenarios = load_demo_scenarios()
    print("scenarios", [s["id"] for s in scenarios])
    assert len(scenarios) >= 4

    demo = DemoController()
    # Run one unsafe + one harmless for speed
    for sid in ["demo-unsafe-refuse", "demo-harmless-help"]:
        print(f"=== run {sid} ===")
        out = demo.run_scenario(sid, mode="playground", save_if_eligible=True)
        j = out["judgment"]
        print("judgment_mode", j.get("judgment_mode"))
        print("eligible", j.get("preference_eligibility"))
        print("retrieved", j.get("retrieved_count"), j.get("retrieval_status", {}).get("backend"))
        print("saved", out.get("saved_preference_id"), out.get("save_blocked_reason"))
        pair = j["preference_pair"]
        assert pair["prompt"]
        assert pair["chosen"] != pair["rejected"] or not j["preference_eligibility"]["eligible"]
        assert "retrieved_references" in pair
        assert "rationale" in pair

    print("demo prefs", demo.preference_count())

    print("=== adapter honesty ===")
    adapter = find_latest_adapter()
    print("adapter", adapter)
    if adapter:
        m = TransformersTarget(adapter_path=adapter)
        m.load()
        info = m.info()
        print(info)
        assert info.get("adapter_loaded") is True
        assert info.get("optimized") is True
        m.unload()
        base = TransformersTarget(adapter_path=None)
        base.load()
        binfo = base.info()
        assert binfo.get("adapter_loaded") is False
        assert binfo.get("optimized") is False
        base.unload()

    print("DEMO SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
