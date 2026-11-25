"""Core demonstration loop: Prompt → Target → Judge → Preference."""

from __future__ import annotations

from typing import Any

import config
from src.judge.agent import JudgeAgent
from src.memory.db import MemoryStore
from src.models.target_llm import OllamaTarget, TransformersTarget, create_target
from src.retrieval.chroma_store import get_safety_store
from src.training.dpo_train import find_latest_adapter


class DemoPipeline:
    def __init__(self, session_id: str = "demo"):
        self.session_id = session_id
        self.memory = MemoryStore()
        self.store = get_safety_store()
        self.adapter_path = find_latest_adapter()

        # Playground quality target (Ollama Qwen by default)
        self.playground_target = create_target(mode="playground")
        # Trainable base + optimized (HF SmolLM2 path)
        self.hf_base_target = TransformersTarget(adapter_path=None)
        self.optimized_target = (
            TransformersTarget(adapter_path=self.adapter_path) if self.adapter_path else None
        )

        self.active_mode = "playground"  # playground | train_base | optimized
        self.judge = JudgeAgent()
        self.judge.tools.session_id = session_id
        self.judge.tools.memory = self.memory
        self.judge.tools.store = self.store

    def set_mode(self, mode: str, ollama_model: str | None = None) -> None:
        if mode not in {"playground", "train_base", "optimized"}:
            raise ValueError(f"Unknown mode: {mode}")
        if mode == "playground" and ollama_model:
            self.playground_target = OllamaTarget(model=ollama_model)
        if mode == "optimized":
            self.reload_optimized()
            if self.optimized_target is None:
                raise RuntimeError("No LoRA adapter available for optimized mode")
            # Probe-load to ensure we never label optimized without a real adapter load
            self.optimized_target.load()
            info = self.optimized_target.info()
            if not info.get("adapter_loaded"):
                raise RuntimeError(
                    "Adapter path exists but adapter did not load; refusing optimized mode"
                )
        self.active_mode = mode

    def get_active_target(self):
        if self.active_mode == "optimized":
            return self.optimized_target or self.hf_base_target
        if self.active_mode == "train_base":
            return self.hf_base_target
        return self.playground_target

    def reload_optimized(self) -> str | None:
        adapter = find_latest_adapter()
        self.adapter_path = adapter
        if adapter:
            self.optimized_target = TransformersTarget(adapter_path=adapter)
            return str(adapter)
        self.optimized_target = None
        return None

    def unload_all_targets(self) -> None:
        for t in (self.playground_target, self.hf_base_target, self.optimized_target):
            if t is not None:
                try:
                    t.unload()
                except Exception:
                    pass

    def generate(
        self,
        prompt: str,
        mode: str | None = None,
        ollama_model: str | None = None,
    ) -> dict[str, Any]:
        if mode:
            self.set_mode(mode, ollama_model=ollama_model)
        model = self.get_active_target()
        history = self.memory.get_history(self.session_id, limit=6)
        response = model.generate(prompt, history=history)
        self.memory.add_turn(self.session_id, "user", prompt)
        self.memory.add_turn(self.session_id, "assistant", response)
        return {
            "prompt": prompt,
            "response": response,
            "mode": self.active_mode,
            "model_info": model.info(),
        }

    def judge_response(
        self, prompt: str, response: str, category: str | None = None
    ) -> dict[str, Any]:
        # Free HF weights before loading the local judge model.
        try:
            self.hf_base_target.unload()
            if self.optimized_target is not None:
                self.optimized_target.unload()
        except Exception:
            pass
        return self.judge.evaluate(
            prompt=prompt,
            target_response=response,
            category=category,
            persist=True,
        )

    def run_once(
        self,
        prompt: str,
        category: str | None = None,
        mode: str | None = None,
    ) -> dict[str, Any]:
        gen = self.generate(prompt, mode=mode)
        judgment = self.judge_response(prompt, gen["response"], category=category)
        return {"generation": gen, **judgment}

    def save_preference_pair(self, pair: dict) -> str:
        return self.judge.save_preference(pair)

    def system_status(self) -> dict[str, Any]:
        return {
            "version": config.APP_VERSION,
            "target_backend_default": config.TARGET_BACKEND,
            "target_ollama_model": config.TARGET_OLLAMA_MODEL,
            "target_hf_model": config.TARGET_HF_MODEL_ID,
            "judge_model": config.JUDGE_MODEL,
            "retrieval": self.store.status,
            "preference_count": self.memory.preference_count(),
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
            "active_mode": self.active_mode,
        }
