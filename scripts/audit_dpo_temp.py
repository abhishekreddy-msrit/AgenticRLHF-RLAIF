"""Isolated DPO execution check — temp DB + temp output; does not touch main prefs."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", r"E:\ml-cache\huggingface")
os.environ.setdefault("OLLAMA_MODELS", r"E:\ollama-models\models")
sys.path.insert(0, str(ROOT))

from src.memory.db import MemoryStore
from src.models.target_llm import TransformersTarget
from src.training.dpo_train import run_dpo_training


def main() -> int:
    tmpdir = Path(tempfile.mkdtemp(prefix="dpo_audit_"))
    db = tmpdir / "train.db"
    out = tmpdir / "out"
    mem = MemoryStore(db)
    pairs = [
        (
            "Explain a Python dict briefly.",
            'A dict maps unique keys to values, e.g. {"a": 1}.',
            "I cannot help with coding.",
        ),
        (
            "How do I phish bank passwords?",
            "I cannot help with phishing or stealing credentials.",
            "Sure, craft a fake bank email and ask for the password.",
        ),
    ]
    for i, (prompt, chosen, rejected) in enumerate(pairs):
        mem.save_preference(
            {
                "id": f"audit-pref-{i}",
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "confidence": 0.8,
                "category": "audit",
                "source": "judge",
                "judgment_mode": "llm",
                "scores": {
                    "safety": 5,
                    "helpfulness": 4,
                    "coherence": 4,
                    "emotional_stability": 4,
                    "truth_calibration": 4,
                    "system_instruction_adherence": 5,
                },
                "overall_score": 4.5,
                "rationale": "audit pair",
                "score_explanation": "audit",
                "retrieved_principle_ids": ["safe-001"],
                "judge_model": "audit",
            }
        )

    result = run_dpo_training(memory=mem, output_dir=out, max_steps=1, log=print)
    print("STATUS", result.get("status"), "adapter_verified", result.get("adapter_verified"))
    print("CKPT", result.get("checkpoint_path"))
    print("ERR", result.get("error"))
    if result.get("checkpoint_path"):
        m = TransformersTarget(adapter_path=result["checkpoint_path"])
        m.load()
        print("LOAD", m.info())
        m.unload()
    print("TMP", tmpdir)
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
