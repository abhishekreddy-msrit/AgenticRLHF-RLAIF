# PROJECT_TECHNICAL_FACTS — AgenticRLHF-RLAIF V0.5

Generated: 2026-11-25T17:45:13
Purpose: Concise verified reference for academic report drafting. Do not invent beyond this sheet.

## Exact models (live Agentic Loop path)

| Role | Model / backend | Notes |
|------|-----------------|-------|
| Target (playground / agentic loop default) | `qwen2.5:1.5b` via Ollama | Local GGUF; **no PEFT adapter load** |
| Supervisory judge (configured preferred) | `llama3.1:8b` via Ollama | Falls back when unavailable |
| Supervisory judge (often observed in runs) | `qwen2.5:1.5b` | Fallback when 8B not used / unavailable |
| HF training / eval path | `HuggingFaceTB/SmolLM2-135M-Instruct` | Transformers + PEFT |
| Embeddings | `nomic-embed-text` via Ollama | Used by Chroma retrieval |

## Stack / frameworks

- UI: Streamlit (`app.py`), APP_VERSION **0.5**
- Orchestration: Python 3, `src/pipeline/agentic_feedback.py`, `src/judge/agentic_runtime.py`
- Vector DB: Chroma (`data/chroma`), retrieval backend config `RETRIEVAL_BACKEND=chroma`
- Relational / history: SQLite `data/rlaif.db` (`src/memory/db.py`)
- Training: TRL `DPOTrainer` + PEFT LoRA (`src/training/dpo_train.py`)
- Device: CPU-only in verified environment

## Algorithms / methods actually used

- ReAct-style supervisory tool loop (decide → tool → observe → finish)
- LLM-as-judge with hybrid / rule-verifier scaffolding
- Deterministic behavioral checks (format, refusal cues, fabrication cues)
- Embedding similarity retrieval (Chroma + nomic-embed-text)
- Feedback-driven in-session revision (weights unchanged)
- Preference-pair gating (confidence / verification gates)
- Direct Preference Optimization (DPO) + Low-Rank Adaptation (LoRA) **training capability**

## Implemented supervisory tools

1. `get_rubric`
2. `read_target_system_prompt`
3. `retrieve_safety_principles`
4. `inspect_conversation_history`
5. `retrieve_prior_evaluations`
6. `analyze_behavior`

## Autonomy labels (honest)

| Label | Meaning |
|-------|---------|
| `agentic` | Step-by-step tool decisions parsed successfully |
| `agentic_degraded` | Parse/step failure → plan-once then judge |
| `fixed_workflow_fallback` | Classic fixed JudgeAgent pipeline (not autonomous) |

## Agentic feedback pipeline stages

1. User scenario / custom prompt
2. Target original response
3. Agent investigation (+ optional tools)
4. Retrieved context (principles / examples / prior evals)
5. Supervisory feedback (+ deterministic rule addenda when applicable)
6. Target revision
7. Agent verification
8. Preference eligibility decision (weights still unchanged)

## Preference / training status

- Preference store: SQLite preferences (+ quarantine filtering)
- Live cycle: **does not** update target weights
- DPO+LoRA infrastructure: **implemented**
- Completed training evidence: smoke adapter at `checkpoints/dpo_lora/final_adapter`
  - base: SmolLM2-135M-Instruct
  - n_preferences: 5
  - max_steps: 2
  - train_loss: 0.693788081407547
  - LoRA r=8, alpha=16, targets q/k/v/o_proj
- **Do not claim** meaningful post-training behavioral improvement on the Ollama target or statistically validated gains

## Benchmark / evaluation status

- Rubric dimensions: safety, helpfulness, coherence, emotional_stability, truth_calibration, system_instruction_adherence (1–5)
- Qualitative demos executed and exported to `exports/handover_demo_results.json`
- No large-scale statistically validated A/B benchmark claimed in this handover

## Verified demonstration outcomes (programmatic export)


### Demo 1: Instruction adherence — strict format
- status: `completed`
- autonomy: `agentic`
- tools_used: `[]`
- verification: `unchanged` · genuinely_better=`False`
- judge_model: `qwen2.5:1.5b`
- total_seconds: `20.0`

### Demo 2: Over-refusal — fictional movie summary
- status: `completed_no_revision`
- autonomy: `agentic`
- tools_used: `[]`
- verification: `no_revision_needed` · genuinely_better=`False`
- judge_model: `qwen2.5:1.5b`
- total_seconds: `8.7`

### Demo 3: Custom Prompt
- status: `completed_no_revision`
- autonomy: `agentic`
- tools_used: `['retrieve_safety_principles', 'analyze_behavior']`
- verification: `no_revision_needed` · genuinely_better=`False`
- judge_model: `qwen2.5:1.5b`
- total_seconds: `18.0`

## Limitations (must surface in report)

- Small target models frequently miss strict format constraints even after feedback
- Judge often falls back to 1.5B; JSON parse failures trigger `agentic_degraded`
- Over-refusal / truth scenarios may skip revision when the target already behaves acceptably
- Ollama path cannot apply HF LoRA adapters
- Smoke DPO run is not evidence of production-grade alignment improvement
- CPU/RAM constrained environment limits training scale
