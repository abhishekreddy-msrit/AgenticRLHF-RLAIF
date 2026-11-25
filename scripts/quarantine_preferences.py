"""
Quarantine legacy / contaminated preferences WITHOUT deleting them.

Marks training_status + quarantine_reason on the main (or specified) DB.
Original chosen/rejected text is preserved.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", r"E:\ml-cache\huggingface")
sys.path.insert(0, str(ROOT))

from src.judge.preference import (
    assess_preference_eligibility,
    chosen_teaches_harm,
    is_cosmetic_pair,
)
from src.memory.db import MemoryStore
import config


def quarantine_reason_for(p: dict) -> str | None:
    reasons = []
    src = str(p.get("source") or "")
    mode = str(p.get("judgment_mode") or "")
    conf = p.get("confidence")
    if src in {"judge_fallback", "smoke_seed", "manual_seed"}:
        reasons.append(f"untrusted_source:{src}")
    if mode == "heuristic_fallback" or src == "judge_fallback":
        reasons.append("heuristic_derived")
    if conf is not None and float(conf) < config.JUDGE_MIN_CONFIDENCE:
        reasons.append(f"low_confidence:{conf}")
    if is_cosmetic_pair(p.get("chosen", ""), p.get("rejected", "")):
        reasons.append("cosmetic_pair")
    if chosen_teaches_harm(p.get("prompt", ""), p.get("chosen", "")):
        reasons.append("chosen_teaches_harm")
    if not (p.get("rationale") or "").strip():
        reasons.append("missing_rationale")
    if not (p.get("retrieved_principle_ids") or []):
        reasons.append("missing_principle_ids")
    # Explicit gate
    gate = assess_preference_eligibility(
        prompt=str(p.get("prompt") or ""),
        chosen=str(p.get("chosen") or ""),
        rejected=str(p.get("rejected") or ""),
        confidence=float(conf or 0.0),
        judgment_mode=mode or ("heuristic_fallback" if src == "judge_fallback" else "llm"),
        verification_ok=True,
        source=src,
        rationale=p.get("rationale"),
        retrieved_principle_ids=p.get("retrieved_principle_ids"),
    )
    if not gate["eligible"]:
        for issue in gate["issues"]:
            if issue not in reasons:
                reasons.append(issue)
    return "; ".join(reasons) if reasons else None


def main() -> int:
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else config.DB_PATH
    mem = MemoryStore(db_path)
    prefs = mem.list_preferences(limit=10000)
    report = {"db": str(db_path), "total": len(prefs), "quarantined": [], "left_unreviewed": []}
    for p in prefs:
        existing = (p.get("training_status") or "").lower()
        if existing == "trusted":
            report.setdefault("kept_trusted", []).append(p["id"])
            continue
        if existing == "quarantined":
            report.setdefault("already_quarantined", []).append(p["id"])
            continue
        reason = quarantine_reason_for(p)
        if reason:
            mem.set_preference_training_status(p["id"], "quarantined", reason)
            report["quarantined"].append({"id": p["id"], "reason": reason, "prompt": (p.get("prompt") or "")[:100]})
        else:
            # Leave as legacy_unreviewed — still not training-eligible until trusted
            mem.set_preference_training_status(p["id"], "legacy_unreviewed", None)
            report["left_unreviewed"].append(p["id"])

    usable = mem.list_training_preferences(limit=10000)
    report["training_eligible_after"] = len(usable)
    out = config.EXPORTS_DIR / "quarantine_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in report.items()}, indent=2))
    print("WROTE", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
