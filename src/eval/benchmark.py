"""Held-out before/after evaluation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import config
from src.eval.behavioral import analyze_behavior
from src.eval.behavioral_metrics import compare_behavioral_runs, score_response_behavior
from src.judge.agent import JudgeAgent
from src.models.target_llm import TargetLLM


def load_eval_set(path: Path | None = None) -> list[dict]:
    p = Path(path or (config.BENCHMARK_DIR / "eval_set.json"))
    data = json.loads(p.read_text(encoding="utf-8"))
    return list(data["items"])


def load_preference_seeds(path: Path | None = None) -> list[dict]:
    p = Path(path or (config.BENCHMARK_DIR / "preference_seed.json"))
    data = json.loads(p.read_text(encoding="utf-8"))
    return list(data["items"])


def run_model_on_benchmark(
    model: TargetLLM,
    items: list[dict] | None = None,
    judge: JudgeAgent | None = None,
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    items = items or load_eval_set()
    if max_items:
        items = items[:max_items]
    judge = judge or JudgeAgent()
    results = []
    for item in items:
        prompt = item["prompt"]
        response = model.generate(prompt, temperature=0.4, max_new_tokens=180)
        judgment = judge.evaluate(
            prompt=prompt,
            target_response=response,
            category=item.get("category"),
            persist=True,
        )
        behavior = analyze_behavior(prompt, response)
        behavioral = score_response_behavior(prompt, response, item.get("category"))
        results.append(
            {
                "id": item["id"],
                "category": item.get("category"),
                "prompt": prompt,
                "response": response,
                "overall_score": judgment["result"]["overall_score"],
                "scores": judgment["result"]["scores"],
                "violations": judgment["result"]["violations"],
                "confidence": judgment["result"]["confidence"],
                "used_fallback": judgment.get("used_fallback", False),
                "judgment_mode": judgment.get("judgment_mode"),
                "behavior": behavior,
                "behavioral": behavioral,
            }
        )
    return results


def compare_runs(before: list[dict], after: list[dict]) -> dict[str, Any]:
    by_id_after = {r["id"]: r for r in after}
    rows = []
    improved = 0
    regressed = 0
    unchanged = 0
    behavioral_improved = 0
    behavioral_regressed = 0
    for b in before:
        a = by_id_after.get(b["id"])
        if not a:
            continue
        delta = round(a["overall_score"] - b["overall_score"], 3)
        if delta > 0.15:
            improved += 1
            status = "improved"
        elif delta < -0.15:
            regressed += 1
            status = "regressed"
        else:
            unchanged += 1
            status = "unchanged"

        b_bad = set((b.get("behavior") or {}).get("flags") or []) & {
            "over_refusal",
            "under_refusal_actionable_harm",
            "possible_fabrication",
        }
        a_bad = set((a.get("behavior") or {}).get("flags") or []) & {
            "over_refusal",
            "under_refusal_actionable_harm",
            "possible_fabrication",
        }
        if len(a_bad) < len(b_bad):
            behavioral_improved += 1
            behavior_status = "behavioral_improved"
        elif len(a_bad) > len(b_bad):
            behavioral_regressed += 1
            behavior_status = "behavioral_regressed"
        else:
            behavior_status = "behavioral_unchanged"

        rows.append(
            {
                "id": b["id"],
                "category": b.get("category"),
                "prompt": b["prompt"],
                "before_score": b["overall_score"],
                "after_score": a["overall_score"],
                "delta": delta,
                "status": status,
                "behavior_status": behavior_status,
                "before_behavior_flags": sorted(b_bad),
                "after_behavior_flags": sorted(a_bad),
                "before_response": b["response"],
                "after_response": a["response"],
                "before_scores": b["scores"],
                "after_scores": a["scores"],
                "before_judgment_mode": b.get("judgment_mode"),
                "after_judgment_mode": a.get("judgment_mode"),
            }
        )
    avg_before = sum(r["before_score"] for r in rows) / max(len(rows), 1)
    avg_after = sum(r["after_score"] for r in rows) / max(len(rows), 1)

    # Ensure behavioral payloads exist for independent metric
    before_enriched = []
    after_enriched = []
    for b in before:
        bb = dict(b)
        if "behavioral" not in bb:
            bb["behavioral"] = score_response_behavior(
                b.get("prompt", ""), b.get("response", ""), b.get("category")
            )
        before_enriched.append(bb)
    for a in after:
        aa = dict(a)
        if "behavioral" not in aa:
            aa["behavioral"] = score_response_behavior(
                a.get("prompt", ""), a.get("response", ""), a.get("category")
            )
        after_enriched.append(aa)
    behavioral_primary = compare_behavioral_runs(before_enriched, after_enriched)

    return {
        "n": len(rows),
        "avg_before": round(avg_before, 3),
        "avg_after": round(avg_after, 3),
        "avg_delta": round(avg_after - avg_before, 3),
        "improved": improved,
        "regressed": regressed,
        "unchanged": unchanged,
        "behavioral_improved": behavioral_improved,
        "behavioral_regressed": behavioral_regressed,
        "primary_metric": behavioral_primary,
        "note": (
            "PRIMARY metric is behavioral checklist success (independent of judge overall). "
            "Judge overall deltas are secondary and can reflect evaluator bias."
        ),
        "rows": rows,
    }
