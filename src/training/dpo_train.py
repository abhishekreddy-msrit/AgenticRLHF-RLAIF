"""Real DPO + LoRA preference optimization for the target model."""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

import config
from src.memory.db import MemoryStore
from src.training.feasibility import assess_training_feasibility


LogFn = Callable[[str], None]


def _default_log(msg: str) -> None:
    print(msg, flush=True)


def preferences_to_dataset(prefs: list[dict]) -> Dataset:
    rows = []
    for p in prefs:
        prompt = p["prompt"]
        # Chat-style prompt string for tiny instruct models
        prompt_text = (
            f"System: {config.TARGET_SYSTEM_PROMPT}\n"
            f"User: {prompt}\n"
            f"Assistant:"
        )
        rows.append(
            {
                "prompt": prompt_text,
                "chosen": p["chosen"],
                "rejected": p["rejected"],
            }
        )
    return Dataset.from_list(rows)


def find_latest_adapter(base: Path | None = None) -> Path | None:
    root = Path(base or config.TRAIN_OUTPUT_DIR)
    if not root.exists():
        return None
    preferred = root / "final_adapter"
    if (preferred / "adapter_config.json").exists():
        return preferred
    # Prefer trainable adapters; skip TRL reference snapshots under /ref/
    candidates = []
    for cfg in root.rglob("adapter_config.json"):
        parent = cfg.parent
        if parent.name.lower() == "ref" or "\\ref\\" in str(parent).lower() or "/ref/" in str(parent).lower():
            continue
        candidates.append(parent)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def run_dpo_training(
    memory: MemoryStore | None = None,
    output_dir: Path | None = None,
    max_steps: int | None = None,
    log: LogFn | None = None,
) -> dict[str, Any]:
    """
    Run an actual DPO+LoRA training job on saved preference pairs.
    Returns a status dict including checkpoint path and logs.
    """
    log = log or _default_log
    memory = memory or MemoryStore()
    output_dir = Path(output_dir or config.TRAIN_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_steps = max_steps or config.TRAIN_MAX_STEPS

    raw_prefs = memory.list_preferences(limit=10000)
    # Strict training gate: quarantine + quality filters (never silent ingest)
    if hasattr(memory, "list_training_preferences"):
        prefs = memory.list_training_preferences(limit=10000)
        skipped = max(0, len(raw_prefs) - len(prefs))
    else:
        from src.judge.preference import assess_stored_preference_for_training

        prefs = []
        skipped = 0
        for p in raw_prefs:
            gate = assess_stored_preference_for_training(p)
            if gate.get("eligible"):
                prefs.append(p)
            else:
                skipped += 1
    _l_pre = (
        f"[train] training-eligible prefs={len(prefs)} "
        f"skipped_or_quarantined={skipped} raw={len(raw_prefs)}"
    )

    report = assess_training_feasibility(n_preferences=len(prefs))
    exp_id = str(uuid4())
    started = datetime.now(timezone.utc).isoformat()
    log_lines: list[str] = []

    def _l(msg: str) -> None:
        log_lines.append(msg)
        log(msg)

    _l(f"[train] feasibility method={report.method} trainable={report.trainable}")
    _l(_l_pre)
    _l(f"[train] preferences usable={len(prefs)} skipped={skipped} raw={len(raw_prefs)}")
    _l(f"[train] base_model={config.TARGET_HF_MODEL_ID}")
    for w in report.warnings:
        _l(f"[warn] {w}")
    for r in report.reasons:
        _l(f"[block] {r}")

    if not report.trainable:
        result = {
            "id": exp_id,
            "status": "blocked",
            "method": "none",
            "error": "; ".join(report.reasons) or "Not trainable",
            "feasibility": report.to_dict(),
            "log_text": "\n".join(log_lines),
            "checkpoint_path": None,
            "created_at": started,
        }
        memory.save_experiment(
            {
                "id": exp_id,
                "name": "dpo_blocked",
                "method": "none",
                "config": report.recommended_config,
                "status": "blocked",
                "metrics": {},
                "checkpoint_path": None,
                "log_text": result["log_text"],
                "created_at": started,
            }
        )
        return result

    if len(prefs) < 2:
        msg = "Need at least 2 preference pairs to run DPO."
        _l(msg)
        result = {
            "id": exp_id,
            "status": "blocked",
            "method": "dpo_lora",
            "error": msg,
            "feasibility": report.to_dict(),
            "log_text": "\n".join(log_lines),
            "checkpoint_path": None,
            "created_at": started,
        }
        memory.save_experiment(
            {
                "id": exp_id,
                "name": "dpo_blocked",
                "method": "dpo_lora",
                "config": report.recommended_config,
                "status": "blocked",
                "log_text": result["log_text"],
                "created_at": started,
            }
        )
        return result

    try:
        _l(f"[train] loading tokenizer/model {config.TARGET_MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(config.TARGET_MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            config.TARGET_MODEL_ID,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        # Reference model for DPO — TRL can create internally if model is Peft; for clarity we pass None
        # and use ref_model=None with peft so TRL disables adapter for reference logprobs.
        lora_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        # Validate target modules exist; fall back if architecture differs
        module_names = {n.split(".")[-1] for n, _ in model.named_modules()}
        if not {"q_proj", "v_proj"}.issubset(module_names):
            # SmolLM / Llama-like usually have these; else try common alternatives
            alt = [m for m in ["c_attn", "q_proj", "v_proj", "query_key_value", "Wqkv"] if m in module_names]
            if not alt:
                # last resort: all Linear leaf names containing 'proj'
                alt = sorted(
                    {
                        n.split(".")[-1]
                        for n, mod in model.named_modules()
                        if n.split(".")[-1].endswith("proj")
                    }
                )[:4]
            _l(f"[train] adjusting LoRA target modules -> {alt}")
            lora_config.target_modules = alt or ["q_proj", "v_proj"]

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        dataset = preferences_to_dataset(prefs)
        _l(f"[train] dataset size={len(dataset)} max_steps={max_steps}")

        training_args = DPOConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=1,
            learning_rate=config.TRAIN_LR,
            max_steps=max_steps,
            logging_steps=1,
            save_steps=max(max_steps, 1),
            bf16=False,
            fp16=False,
            remove_unused_columns=False,
            max_length=config.TRAIN_MAX_LENGTH,
            beta=config.DPO_BETA,
            report_to=[],
            optim="adamw_torch",
            warmup_steps=0,
            dataloader_pin_memory=False,
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        _l("[train] starting DPOTrainer.train() — real preference optimization")
        train_result = trainer.train()
        metrics = dict(train_result.metrics or {})
        _l(f"[train] finished metrics={json.dumps(metrics)}")

        adapter_dir = output_dir / "final_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        meta = {
            "method": "DPO + LoRA",
            "base_model": config.TARGET_MODEL_ID,
            "n_preferences": len(prefs),
            "max_steps": max_steps,
            "metrics": metrics,
            "created_at": started,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        # Verify adapter artifacts exist before claiming completion
        adapter_cfg = adapter_dir / "adapter_config.json"
        adapter_weights = adapter_dir / "adapter_model.safetensors"
        if not adapter_cfg.exists() or not adapter_weights.exists():
            raise RuntimeError(
                f"Adapter save incomplete: cfg={adapter_cfg.exists()} "
                f"weights={adapter_weights.exists()}"
            )
        cfg = json.loads(adapter_cfg.read_text(encoding="utf-8"))
        meta["adapter_base_model"] = cfg.get("base_model_name_or_path")
        meta["adapter_verified"] = True
        (adapter_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        _l(f"[train] saved adapter -> {adapter_dir}")
        _l(f"[train] adapter base_model={cfg.get('base_model_name_or_path')}")

        result = {
            "id": exp_id,
            "status": "completed",
            "method": "dpo_lora",
            "method_label": "DPO + LoRA (TRL DPOTrainer)",
            "checkpoint_path": str(adapter_dir),
            "metrics": metrics,
            "feasibility": report.to_dict(),
            "log_text": "\n".join(log_lines),
            "n_preferences": len(prefs),
            "n_skipped": skipped,
            "adapter_verified": True,
            "created_at": started,
        }
        memory.save_experiment(
            {
                "id": exp_id,
                "name": "dpo_lora",
                "method": "dpo_lora",
                "config": report.recommended_config,
                "status": "completed",
                "metrics": metrics,
                "checkpoint_path": str(adapter_dir),
                "log_text": result["log_text"],
                "created_at": started,
            }
        )
        # free memory
        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    except Exception as e:
        tb = traceback.format_exc()
        _l(f"[error] {e}")
        _l(tb)
        result = {
            "id": exp_id,
            "status": "failed",
            "method": "dpo_lora",
            "error": str(e),
            "checkpoint_path": None,
            "feasibility": report.to_dict(),
            "log_text": "\n".join(log_lines),
            "created_at": started,
        }
        memory.save_experiment(
            {
                "id": exp_id,
                "name": "dpo_lora_failed",
                "method": "dpo_lora",
                "config": report.recommended_config,
                "status": "failed",
                "metrics": {"error": str(e)},
                "checkpoint_path": None,
                "log_text": result["log_text"],
                "created_at": started,
            }
        )
        return result


def load_optimized_target(adapter_path: str | Path | None = None):
    from src.models.target_llm import TargetLLM

    path = Path(adapter_path) if adapter_path else find_latest_adapter()
    return TargetLLM(adapter_path=path)
