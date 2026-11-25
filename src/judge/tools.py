"""Judge tools as simple Python functions (harness engineering — no MCP required)."""

from __future__ import annotations

from typing import Any

import config
from src.judge.rubric import rubric_summary
from src.memory.db import MemoryStore
from src.retrieval.chroma_store import ChromaSafetyStore, get_safety_store


class JudgeTools:
    def __init__(
        self,
        memory: MemoryStore | None = None,
        store: ChromaSafetyStore | None = None,
        session_id: str = "default",
    ):
        self.memory = memory or MemoryStore()
        self.store = store or get_safety_store()
        self.session_id = session_id
        self.call_log: list[dict[str, Any]] = []

    def _log(self, name: str, args: dict, result_preview: Any) -> None:
        self.call_log.append(
            {
                "tool": name,
                "args": args,
                "result_preview": result_preview,
            }
        )

    def reset_log(self) -> None:
        self.call_log = []

    def get_rubric(self) -> dict:
        out = rubric_summary()
        self._log("get_rubric", {}, {"dimensions": out["dimensions"]})
        return out

    def read_target_system_prompt(self) -> str:
        prompt = config.TARGET_SYSTEM_PROMPT
        self._log("read_target_system_prompt", {}, prompt[:120] + "...")
        return prompt

    def retrieve_safety_principles(self, prompt: str, response: str, k: int | None = None) -> list[dict]:
        k = k or config.RETRIEVAL_TOP_K
        hits = self.store.retrieve_for_prompt_response(prompt, response, k=k)
        self._log(
            "retrieve_safety_principles",
            {"k": k, "backend": self.store.backend},
            [
                {
                    "id": h.get("id"),
                    "title": h.get("title"),
                    "category": h.get("category"),
                    "score": h.get("score"),
                    "backend": h.get("backend"),
                }
                for h in hits
            ],
        )
        return hits

    def inspect_conversation_history(self, limit: int = 8) -> list[dict]:
        hist = self.memory.get_history(self.session_id, limit=limit)
        self._log("inspect_conversation_history", {"limit": limit}, f"{len(hist)} turns")
        return hist

    def retrieve_prior_evaluations(self, limit: int = 5) -> list[dict]:
        evals = self.memory.list_evaluations(limit=limit)
        preview = [
            {
                "id": e["id"],
                "overall_score": e.get("overall_score"),
                "prompt": (e.get("prompt") or "")[:80],
            }
            for e in evals
        ]
        self._log("retrieve_prior_evaluations", {"limit": limit}, preview)
        return evals

    def save_evaluation(self, result: dict) -> str:
        eid = self.memory.save_evaluation(result)
        try:
            self.store.add_evaluation_case(result)
        except Exception:
            pass
        self._log("save_evaluation", {"id": eid}, "saved")
        return eid

    def create_preference_pair(self, pair: dict) -> str:
        pid = self.memory.save_preference(pair)
        try:
            self.store.add_preference_example(pair)
        except Exception:
            pass
        self._log("create_preference_pair", {"id": pid}, "saved")
        return pid

    def retrieval_status(self) -> dict:
        return self.store.status
