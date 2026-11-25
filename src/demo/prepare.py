"""Demo Mode helpers: scenarios, isolated demo DB, reproducible workflow."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import config
from src.judge.agent import JudgeAgent
from src.memory.db import MemoryStore
from src.models.target_llm import TransformersTarget
from src.pipeline.loop import DemoPipeline
from src.training.dpo_train import find_latest_adapter, run_dpo_training


DEMO_DB_PATH = config.DATA_DIR / "demo_rlaif.db"
DEMO_SCENARIOS_PATH = config.BENCHMARK_DIR / "demo_scenarios.json"


def load_demo_scenarios(path: Path | None = None) -> list[dict[str, Any]]:
    p = Path(path or DEMO_SCENARIOS_PATH)
    data = json.loads(p.read_text(encoding="utf-8"))
    return list(data.get("scenarios") or [])


def reset_demo_store(backup: bool = True) -> dict[str, Any]:
    """
    Reset ONLY the demo SQLite database.
    Never deletes the main project DB, checkpoints, or chroma store.
    """
    info: dict[str, Any] = {"demo_db": str(DEMO_DB_PATH), "reset": False, "backup": None}
    if DEMO_DB_PATH.exists():
        if backup:
            bak = DEMO_DB_PATH.with_suffix(".db.bak")
            shutil.copy2(DEMO_DB_PATH, bak)
            info["backup"] = str(bak)
        DEMO_DB_PATH.unlink()
        info["reset"] = True
    # Recreate empty schema
    MemoryStore(DEMO_DB_PATH)
    info["preference_count"] = 0
    return info


class DemoController:
    """Faculty-facing orchestration over an isolated demo namespace."""

    def __init__(self):
        self.memory = MemoryStore(DEMO_DB_PATH)
        self.pipeline = DemoPipeline(session_id="faculty_demo")
        # Isolate persistence for demo runs
        self.pipeline.memory = self.memory
        self.pipeline.judge.tools.memory = self.memory
        self.scenarios = load_demo_scenarios()
        self.last_runs: dict[str, dict[str, Any]] = {}

    def scenario_by_id(self, scenario_id: str) -> dict[str, Any] | None:
        for s in self.scenarios:
            if s["id"] == scenario_id:
                return s
        return None

    def run_scenario(
        self,
        scenario_id: str,
        mode: str = "playground",
        save_if_eligible: bool = True,
    ) -> dict[str, Any]:
        scenario = self.scenario_by_id(scenario_id)
        if not scenario:
            raise ValueError(f"Unknown scenario: {scenario_id}")

        gen = self.pipeline.generate(scenario["prompt"], mode=mode)
        judgment = self.pipeline.judge_response(
            scenario["prompt"],
            gen["response"],
            category=scenario.get("category"),
        )
        saved_id = None
        save_blocked_reason = None
        eligibility = judgment.get("preference_eligibility") or {}
        if save_if_eligible:
            if eligibility.get("eligible"):
                saved_id = self.pipeline.save_preference_pair(judgment["preference_pair"])
            else:
                save_blocked_reason = eligibility.get("reason") or "Preference not eligible"

        out = {
            "scenario": scenario,
            "generation": gen,
            "judgment": judgment,
            "saved_preference_id": saved_id,
            "save_blocked_reason": save_blocked_reason,
        }
        self.last_runs[scenario_id] = out
        return out

    def run_all_train_scenarios(self, mode: str = "playground") -> list[dict[str, Any]]:
        results = []
        for s in self.scenarios:
            if s.get("train_split", True):
                results.append(self.run_scenario(s["id"], mode=mode, save_if_eligible=True))
        return results

    def preference_count(self) -> int:
        return self.memory.preference_count()

    def train_demo(self, max_steps: int = 4, log=None) -> dict[str, Any]:
        self.pipeline.unload_all_targets()
        return run_dpo_training(memory=self.memory, max_steps=max_steps, log=log)

    def compare_demo_scenarios(
        self,
        max_scenarios: int | None = None,
        use_held_out_only: bool = False,
    ) -> dict[str, Any]:
        """
        Before/after on demo prompts using HF base vs HF+adapter.
        Honest: skips optimized run if adapter is not loadable.
        """
        scenarios = self.scenarios
        if use_held_out_only:
            scenarios = [s for s in scenarios if not s.get("train_split", True)]
        if max_scenarios:
            scenarios = scenarios[:max_scenarios]

        adapter = find_latest_adapter()
        base = TransformersTarget(adapter_path=None)
        items = [
            {"id": s["id"], "category": s.get("category"), "prompt": s["prompt"]}
            for s in scenarios
        ]

        before = []
        after = []
        judge = self.pipeline.judge

        for item in items:
            prompt = item["prompt"]
            b_resp = base.generate(prompt, temperature=0.3, max_new_tokens=160)
            b_j = judge.evaluate(prompt, b_resp, category=item.get("category"), persist=False)
            from src.eval.behavioral_metrics import score_response_behavior

            before.append(
                {
                    "id": item["id"],
                    "category": item.get("category"),
                    "prompt": prompt,
                    "response": b_resp,
                    "overall_score": b_j["result"]["overall_score"],
                    "scores": b_j["result"]["scores"],
                    "violations": b_j["result"]["violations"],
                    "confidence": b_j["result"]["confidence"],
                    "judgment_mode": b_j.get("judgment_mode"),
                    "behavioral": score_response_behavior(
                        prompt, b_resp, item.get("category")
                    ),
                }
            )

        optimized_ready = False
        adapter_info = None
        if adapter:
            opt = TransformersTarget(adapter_path=adapter)
            try:
                opt.load()
                adapter_info = opt.info()
                optimized_ready = bool(adapter_info.get("adapter_loaded"))
            except Exception as e:
                adapter_info = {"adapter_loaded": False, "load_error": str(e)}
                optimized_ready = False

            if optimized_ready:
                for item in items:
                    prompt = item["prompt"]
                    a_resp = opt.generate(prompt, temperature=0.3, max_new_tokens=160)
                    a_j = judge.evaluate(
                        prompt, a_resp, category=item.get("category"), persist=False
                    )
                    from src.eval.behavioral_metrics import score_response_behavior

                    after.append(
                        {
                            "id": item["id"],
                            "category": item.get("category"),
                            "prompt": prompt,
                            "response": a_resp,
                            "overall_score": a_j["result"]["overall_score"],
                            "scores": a_j["result"]["scores"],
                            "violations": a_j["result"]["violations"],
                            "confidence": a_j["result"]["confidence"],
                            "judgment_mode": a_j.get("judgment_mode"),
                            "behavioral": score_response_behavior(
                                prompt, a_resp, item.get("category")
                            ),
                        }
                    )
            try:
                opt.unload()
            except Exception:
                pass
        try:
            base.unload()
        except Exception:
            pass

        from src.eval.benchmark import compare_runs

        comparison = compare_runs(before, after) if after else None
        return {
            "before": before,
            "after": after,
            "comparison": comparison,
            "optimized_ready": optimized_ready,
            "adapter_path": str(adapter) if adapter else None,
            "adapter_info": adapter_info,
            "benchmark_note": (
                "Small demonstration benchmark only. Same judge scores both sides "
                "(evaluator bias possible). Do not treat as a general safety claim."
            ),
        }
