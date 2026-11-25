# PROJECT_TECHNICAL_FACTS v2 — AgenticRLHF-RLAIF

Generated: 2026-11-25T18:02:08
Purpose: Evidence-bounded quick reference for Pawan’s final academic report drafting.
Companion: `AgenticRLHF_Application_Technical_Handover_v2.docx`

## Status labels (do not interchange)

| Label | Meaning |
|-------|---------|
| Implemented | Present in code/UI |
| Demonstrated | Shown in an executed cycle (UI and/or export) |
| Tested | Exercised by smoke/scripted tests |
| Experimentally evaluated | Has quantitative comparison artifacts |
| Preliminary | Exists but insufficient for strong claims |
| Not yet established | Proposed / future |

## Models

| Role | Exact ID | Path | Status |
|------|----------|------|--------|
| Target (Agentic Loop / Playground default) | `qwen2.5:1.5b` | Ollama | Implemented + Demonstrated |
| Judge preferred (config) | `llama3.1:8b` | Ollama | Implemented |
| Judge often observed in demos | `qwen2.5:1.5b` | Ollama fallback | Demonstrated |
| HF train/eval base | `HuggingFaceTB/SmolLM2-135M-Instruct` | Transformers | Implemented + Tested (smoke) |
| Embeddings | `nomic-embed-text` | Ollama → Chroma | Implemented + Demonstrated |

## Stack

Streamlit UI · Python · Ollama · Transformers/PEFT/TRL · Chroma · SQLite · CPU-only verified env · APP_VERSION 0.5

## Tools (implemented)

`get_rubric`, `read_target_system_prompt`, `retrieve_safety_principles`, `inspect_conversation_history`, `retrieve_prior_evaluations`, `analyze_behavior`

## Autonomy modes

`agentic` · `agentic_degraded` · `fixed_workflow_fallback`

## Live vs offline (critical)

- Live cycle: investigate → feedback → revise → verify → optional preference save. **Weights unchanged.**
- Offline: DPO + LoRA on HF SmolLM2 using eligible preference pairs.

## Training evidence (smoke only)

- Path: `checkpoints/dpo_lora/final_adapter`
- n_preferences: 5
- max_steps: 2
- train_loss: 0.693788081407547
- LoRA: r=8, alpha=16, modules q/k/v/o_proj
- Status: **Preliminary** infrastructure proof — **not** meaningful alignment improvement

## Baseline vs optimized

- Baseline = HF SmolLM2 **without** adapter
- Optimized = HF SmolLM2 **with** loaded LoRA adapter (not Ollama)
- Evaluation UI/code exists (`src/eval/benchmark.py`, Evaluation tab)
- Completed quantitative baseline↔optimized score table for this handover: **not verified from available evidence**
- Adapter load verification exists in audit (`adapter_loaded=true`) — load ≠ performance gain

## Demonstrated experiment outcomes (authoritative export)

See matrix in handover v2. Summary:

1. Instruction adherence — revision occurred; verification **unchanged**
2. Over-refusal (Zorblax) — **no_revision_needed**
3. Custom Elbonia — tools used; **no_revision_needed**
4. UI Emotional stability — **no_revision_needed**
5. UI Instruction adherence (live) — **agentic_degraded**, verify **unchanged**

## Claim boundaries

### Acceptable
- The framework implements and demonstrates an agentic supervisory feedback loop with tools, retrieval, revision, and verification.
- Preference gates can block pairs; live revision does not update weights.
- A smoke DPO+LoRA run produced a loadable adapter.

### Not acceptable without further evidence
- “X% safer / more aligned after DPO”
- “Optimized model outperforms baseline on held-out eval”
- “Every cycle is fully agentic”
- “MCP/LangGraph is part of the implemented system”

## Sources

- `exports/handover_demo_results.json`
- `exports/handover_v2_evidence_pack.json`
- `checkpoints/dpo_lora/final_adapter/train_meta.json`
- `resource material/application_screenshots/` (originals preserved)
