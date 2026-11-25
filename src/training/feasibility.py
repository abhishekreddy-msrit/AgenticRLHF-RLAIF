"""Honest hardware / dependency feasibility checks for preference optimization."""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from typing import Any

import torch

import config


@dataclass
class FeasibilityReport:
    trainable: bool
    method: str
    reasons: list[str]
    warnings: list[str]
    hardware: dict[str, Any]
    recommended_config: dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


def assess_training_feasibility(n_preferences: int = 0) -> FeasibilityReport:
    reasons: list[str] = []
    warnings: list[str] = []

    cuda = torch.cuda.is_available()
    vram_gb = 0.0
    if cuda:
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        except Exception:
            vram_gb = 0.0

    # RAM estimate
    ram_total_gb = None
    ram_avail_gb = None
    try:
        import psutil

        vm = psutil.virtual_memory()
        ram_total_gb = round(vm.total / 1e9, 1)
        ram_avail_gb = round(vm.available / 1e9, 1)
    except Exception:
        pass

    disk_free_gb = round(shutil.disk_usage(config.ROOT).free / 1e9, 1)

    deps_ok = True
    for mod in ("transformers", "peft", "trl", "datasets", "accelerate"):
        try:
            __import__(mod)
        except Exception as e:
            deps_ok = False
            reasons.append(f"Missing/broken dependency '{mod}': {e}")

    hardware = {
        "cuda": cuda,
        "vram_gb": round(vram_gb, 2),
        "torch_version": torch.__version__,
        "device_default": config.DEVICE,
        "ram_total_gb": ram_total_gb,
        "ram_available_gb": ram_avail_gb,
        "disk_free_gb": disk_free_gb,
        "target_model": config.TARGET_MODEL_ID,
        "n_preferences": n_preferences,
    }

    method = "dpo_lora_cpu"
    trainable = deps_ok

    if not cuda:
        warnings.append(
            "No GPU detected. Running real DPO+LoRA on CPU with a tiny model "
            f"({config.TARGET_MODEL_ID}). Expect slow training."
        )
    if ram_avail_gb is not None and ram_avail_gb < 2.5:
        warnings.append(
            f"Low available RAM ({ram_avail_gb} GB). Close other apps before training."
        )
    if n_preferences < 4:
        warnings.append(
            f"Only {n_preferences} preference pairs. Training can run, but results will be weak; "
            "collect at least 4–8 pairs for a clearer demo."
        )
    if disk_free_gb < 2:
        trainable = False
        reasons.append(f"Insufficient disk free space ({disk_free_gb} GB).")

    if not trainable and not reasons:
        reasons.append("Environment checks failed.")

    recommended = {
        "method": "DPO + LoRA (PEFT) via TRL DPOTrainer",
        "model": config.TARGET_MODEL_ID,
        "device": "cpu" if not cuda else "cuda",
        "lora_r": config.LORA_R,
        "lora_alpha": config.LORA_ALPHA,
        "batch_size": config.TRAIN_BATCH_SIZE,
        "max_steps": config.TRAIN_MAX_STEPS,
        "max_length": config.TRAIN_MAX_LENGTH,
        "learning_rate": config.TRAIN_LR,
        "beta": config.DPO_BETA,
        "note": "This is real parameter-efficient preference optimization, not prompt rewriting.",
    }

    if not trainable:
        method = "unavailable"

    return FeasibilityReport(
        trainable=trainable,
        method=method,
        reasons=reasons,
        warnings=warnings,
        hardware=hardware,
        recommended_config=recommended,
    )
