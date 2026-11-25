# AgenticRLHF-RLAIF

**Version:** 0.5  
**Status:** Experimental research prototype (local, CPU-oriented)

AgenticRLHF-RLAIF is a local experimental framework for studying **tool-augmented supervisory evaluation** of language-model responses in a reinforcement-learning-from-AI-feedback (RLAIF) setting. A **Target LLM** generates answers; a separate **Supervisory Agent** may select tools, retrieve safety context, produce corrective feedback, trigger an in-session revision, and verify whether behavior improved. Eligible original/revised pairs can be stored for offline **Direct Preference Optimization (DPO)** with **LoRA** adapters on a small Hugging Face model.

The interactive revision cycle does **not** update model weights. Weight adaptation is a separate offline training path.

---

## 1. Motivation and problem statement

Small instruct models frequently fail under practical alignment pressures: format constraints, over-refusal of harmless requests, fabrication under factual pressure, and instability under hostile user tone. Conventional LLM-as-a-judge pipelines often score responses in a single opaque pass, which makes failures difficult to audit. Full fine-tuning is also costly on student and laboratory hardware.

This project explores a bounded alternative: a **transparent supervisory loop** in which a judge agent can call inspection and retrieval tools before issuing feedback, combined with an optional **preference harvesting** path for offline DPO+LoRA. The goal is methodological clarity and inspectability, not a claim of state-of-the-art safety performance.

---

## 2. Research objectives and intended contribution

1. Implement a **dual-model** architecture that cleanly separates the Target LLM from a Supervisory Agent.
2. Make supervisory decisions **observable** through tool traces, retrieval hits, and labeled autonomy modes.
3. Support **in-session correction** (feedback → revision → verification) without conflating it with persistent learning.
4. Gate preference pairs before training to reduce noisy DPO signals.
5. Provide a runnable Streamlit application and scripts suitable for demonstration and controlled experiments on CPU-only hardware.

**Non-claims.** This repository does not presently establish statistically significant alignment gains, human-validated superiority over baselines, publication acceptance, or production readiness. Smoke-scale DPO evidence shows that the training pipeline executes; it does **not** demonstrate meaningful behavioral improvement.

---

## 3. System architecture

### 3.1 Separation of roles

| Role | Responsibility | Runtime (default) | Model (default) |
|------|----------------|-------------------|-----------------|
| Target LLM | Generates original and revised answers. Does **not** select supervisory tools or query Chroma. | Ollama | `qwen2.5:1.5b` |
| Supervisory Agent / Judge | Investigates responses, may call tools, retrieves principles, produces feedback, verifies revisions. | Ollama | Preferred `llama3.1:8b`; fallbacks include `qwen2.5:1.5b` |
| Embeddings | Vectorize principles/examples for retrieval | Ollama | `nomic-embed-text` |
| Trainable / “optimized” path | Offline DPO+LoRA and HF evaluation | Hugging Face Transformers + PEFT + TRL | `HuggingFaceTB/SmolLM2-135M-Instruct` |

Hugging Face LoRA adapters **cannot** be loaded into Ollama GGUF models. Interactive demos and offline optimization therefore use **distinct** model stacks.

### 3.2 Live inference / feedback loop versus offline training

```
LIVE (interactive; weights unchanged)
  User prompt
    → Target generates initial response
    → Supervisory Agent: tool decisions / retrieval / analysis
    → Feedback to Target (optional rule-verifier addenda)
    → Target revises (if triggered)
    → Supervisory verification
    → Preference eligibility gate → optional SQLite preference record

OFFLINE (separate process)
  Eligible preference pairs
    → DPO objective + LoRA adapters on SmolLM2
    → Adapter checkpoint
    → Optional HF baseline vs HF+adapter evaluation
```

MCP and LangGraph are **not** part of the implemented runtime. They are listed under future work only.

### 3.3 Agentic feedback-loop design

The supervisory runtime (`src/judge/agentic_runtime.py`, orchestrated by `src/pipeline/agentic_feedback.py`) follows a bounded ReAct-style loop: the agent repeatedly decides to `use_tool` or `finish`, accumulates observations, then emits judgment and feedback. Autonomy is labeled honestly:

| Mode | Meaning |
|------|---------|
| `agentic` | Step-by-step tool loop completed with an LLM `finish` decision |
| `agentic_degraded` | Structured JSON failures caused a plan-once then judge recovery; tools may still have been LLM-selected |
| `fixed_workflow_fallback` | Classic non-autonomous judge workflow; not claimed as agentic autonomy |

Verification outcomes observed in demos include `corrected`, `improved`, `unchanged`, `regressed`, and `no_revision_needed`. Unchanged or skipped revision outcomes are reported as such; they are not relabeled as success.

---

## 4. Supervisory tools, retrieval, and memory

### 4.1 Available tools

Registered in `src/judge/tool_registry.py` and executed through real tool backends (results are not fabricated):

| Tool | Purpose |
|------|---------|
| `get_rubric` | Evaluation dimensions and scoring rules |
| `read_target_system_prompt` | Read-only inspection of target instructions |
| `retrieve_safety_principles` | Semantic retrieval of principles/examples from Chroma |
| `inspect_conversation_history` | Recent session turns |
| `retrieve_prior_evaluations` | Prior evaluation summaries |
| `analyze_behavior` | Deterministic cues: refusal, over/under-refusal, fabrication, format constraints |

The Target LLM does not call these tools. Tool use belongs exclusively to the supervisory path.

### 4.2 Retrieval (ChromaDB)

- **Store:** Chroma under `data/chroma/` (local, regenerable; ignored by Git).
- **Corpus seed:** `data/safety/principles.json`, `data/safety/examples.json`.
- **Embeddings:** Ollama model `nomic-embed-text` (configurable via `OLLAMA_EMBED_MODEL`).
- **Top-k:** `RETRIEVAL_TOP_K` (default 4).
- **Initialization:** `python scripts/init_retrieval.py` rebuilds/refreshes the corpus.

Retrieved documents are passed into the supervisory observation context so the judge can ground feedback in principles. Retrieval influence on scores is not automatically measured as an ablation in the current release.

### 4.3 Memory / persistence (SQLite)

`src/memory/db.py` persists conversations, evaluations, and preference records in SQLite (`data/rlaif.db` for normal use; Demo Mode can use `data/demo_rlaif.db`). Databases are local runtime artifacts and are Git-ignored.

---

## 5. Behavioral evaluation, revision, and preferences

### 5.1 Evaluation dimensions

Configured rubric dimensions (`config.RUBRIC_DIMENSIONS`): safety, helpfulness, coherence, emotional stability, truth calibration, and system-instruction adherence (scores typically on a 1–5 scale). Behavioral/deterministic checks are treated as primary evidence for verification in the agentic cycle.

### 5.2 Feedback and revision

When the supervisory path finds an actionable issue, it produces `feedback_to_target`. A deterministic rule verifier may append format-related addenda. The Target then regenerates under that feedback text. The Target still has no tool access during revision.

### 5.3 Preference eligibility

Not every original/revised pair is accepted for training. Gates in `src/judge/preference.py` and the agentic pipeline check verification outcome, confidence thresholds, cosmetic/meta-judgment filters, and related trust criteria. Blocked pairs are an integrity feature, not a UI failure.

Preference semantics used for DPO: **chosen = revised**, **rejected = original**, when a pair is eligible and saved.

---

## 6. DPO and LoRA (offline)

### 6.1 Direct Preference Optimization

DPO trains a policy to prefer a chosen completion over a rejected completion for the same prompt without a separate reward model. Implementation: TRL `DPOTrainer` in `src/training/dpo_train.py`, with preference rows drawn from stored SQLite pairs. Default \(\beta\) (`DPO_BETA`) is `0.1`.

### 6.2 LoRA

Low-Rank Adaptation updates small adapter matrices instead of full model weights. Project defaults: `LORA_R=8`, `LORA_ALPHA=16`, attention projections `q/k/v/o_proj` on `HuggingFaceTB/SmolLM2-135M-Instruct`.

DPO is the preference objective; LoRA is the parameter-efficient adaptation method. They are complementary.

### 6.3 Baseline versus optimized (HF path)

| Term | Meaning in this repository |
|------|----------------------------|
| Baseline | SmolLM2-135M-Instruct **without** adapter |
| Optimized | Same HF base **with** a successfully loaded PEFT LoRA adapter |

Evaluation helpers live in `src/eval/benchmark.py` and the Streamlit **Evaluation** tab, using prompts such as `data/benchmark/eval_set.json`.

**Evidence boundary.** A smoke training run in this workspace used **5 preferences**, **2 max steps**, and reported `train_loss ≈ 0.6938`, with adapter load verification. That establishes pipeline executability. It does **not** establish meaningful post-training behavioral improvement or a verified quantitative baseline-versus-optimized superiority table for publication claims.

---

## 7. Application interface

Entry point: `app.py` (Streamlit), `APP_VERSION = 0.5`.

| Tab | Function |
|-----|----------|
| Agentic Loop | Staged live cycle: scenario/custom prompt → investigate → revise → verify → preference gate |
| Demo Mode | Curated demos; demo DB reset clears only `data/demo_rlaif.db` |
| Playground | Ad-hoc generate / judge |
| Judge | Inspect scores, critique, retrieval, tools |
| Preference Data | Browse/export preference pairs |
| Training | Start DPO+LoRA; show checkpoint path |
| Evaluation | HF baseline / HF+LoRA evaluation hooks; retrieval refresh |

---

## 8. Technology stack

| Layer | Technology |
|-------|------------|
| UI | Streamlit |
| Language | Python 3 |
| Local LLM runtime | Ollama |
| HF training/eval | PyTorch, Transformers, PEFT, TRL, Datasets |
| Vector store | ChromaDB |
| Persistence | SQLite |
| Config | `config.py` + `.env` (from `.env.example`) |

See `requirements.txt` for pinned dependency ranges.

---

## 9. Installation and environment

### 9.1 Prerequisites

- Python 3.10+ recommended
- [Ollama](https://ollama.com) installed and running locally
- Sufficient disk for Hugging Face caches if you use the training/eval path
- CPU is supported (`DEVICE=cpu` by default); GPU is optional

### 9.2 Clone and install

```powershell
git clone https://github.com/abhishekreddy-msrit/AgenticRLHF-RLAIF.git
cd AgenticRLHF-RLAIF

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` as needed. Do not commit `.env`.

### 9.3 Ollama models

Pull at least:

```powershell
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text
```

Strongly recommended for the preferred judge:

```powershell
ollama pull llama3.1:8b
```

If the preferred judge is unavailable, the runtime falls back (often to `qwen2.5:1.5b`). Document the **observed** judge model when reporting experiments; do not assume the configured preferred model was used.

### 9.4 Hugging Face model (training / optimized eval only)

The first DPO or HF evaluation run downloads `HuggingFaceTB/SmolLM2-135M-Instruct` into your HF cache (`HF_HOME` if set).

### 9.5 Initialize retrieval

With Ollama running and `nomic-embed-text` available:

```powershell
python scripts\init_retrieval.py
```

### 9.6 Important environment variables

Defined in `.env.example` / `config.py`:

| Variable | Default / example | Role |
|----------|-------------------|------|
| `TARGET_BACKEND` | `ollama` | Interactive target backend |
| `TARGET_OLLAMA_MODEL` | `qwen2.5:1.5b` | Interactive target model |
| `TARGET_HF_MODEL_ID` | `HuggingFaceTB/SmolLM2-135M-Instruct` | HF train/eval base |
| `JUDGE_MODEL` | `llama3.1:8b` | Preferred supervisory model |
| `JUDGE_FALLBACK_MODELS` | `qwen2.5:1.5b,tinyllama` | Fallbacks |
| `JUDGE_MIN_CONFIDENCE` | `0.52` | Preference confidence gate |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama endpoint |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `RETRIEVAL_TOP_K` | `4` | Retrieval depth |
| `DEVICE` | `cpu` | Torch device |
| `LORA_R` / `LORA_ALPHA` | `8` / `16` | LoRA hyperparameters |
| `DPO_BETA` | `0.1` | DPO temperature parameter |
| `TRAIN_MAX_STEPS` | `12` | Training step budget |
| `HF_HOME` | (optional) | HF cache root |

`scripts/run_app.ps1` sets workspace-local cache paths suitable for the original development machine; adjust paths for your environment.

---

## 10. Running the application

```powershell
# Option A
.\scripts\run_app.ps1

# Option B
streamlit run app.py

# Optional fixed port used in some internal notes
streamlit run app.py --server.port 8510
```

Open the URL printed by Streamlit (commonly `http://127.0.0.1:8501/` or `:8510`).

---

## 11. Demonstration procedure (agentic feedback cycle)

1. Confirm Ollama is running (`ollama list`) and required models are present.
2. Launch the app and open the **Agentic Loop** tab (Demo Mode is an alternative for curated demos).
3. Select a scenario (for example, instruction-adherence / format, emotional stability, over-refusal, or truth-pressure) **or** enter a custom prompt. Confirm the prompt preview.
4. Run the agentic cycle.
5. Inspect **Stage: original Target response** (model id, scores/flags if shown).
6. Inspect the **supervisory investigation / tool trace**: decisions, tools used, degraded-mode notices if any.
7. Inspect **retrieved principles/context** when retrieval ran.
8. Inspect **supervisory feedback** sent to the Target (including any rule-verifier addenda).
9. If revision triggered, inspect **original vs revised** side by side.
10. Inspect **verification** (`corrected` / `improved` / `unchanged` / `regressed` / `no_revision_needed`) and whether the revision was judged genuinely better.
11. Inspect the **preference eligibility** panel: eligible pairs may be stored; blocked pairs cite a reason (for example, verification unchanged).
12. State explicitly during demos: **model weights were not updated** by this interactive cycle.

Optional follow-ups in other tabs:

- **Preference Data** — browse saved chosen/rejected pairs.
- **Training** — run DPO+LoRA only when you intentionally want offline optimization; treat small-N runs as smoke tests.
- **Evaluation** — HF baseline vs HF+adapter only after an adapter exists and load verification succeeds.

---

## 12. Testing and verification

Lightweight checks (Ollama required for most agentic tests; do not retrain as part of routine verification):

```powershell
python scripts\test_judge_rules.py
python scripts\test_ui_render_safety.py
python scripts\smoke_agentic_v05.py
```

Additional scripts exist under `scripts/` for judge benchmarks, preference quarantine, and demos. Prefer reading script headers before long runs. Avoid deleting `checkpoints/` artifacts unless you intend to discard local adapters.

---

## 13. Current implementation status

| Component | Status |
|-----------|--------|
| Target LLM (Ollama) | Implemented and demonstrated |
| Supervisory agent with tools | Implemented and demonstrated |
| Chroma retrieval + embeddings | Implemented and demonstrated |
| SQLite preference/history store | Implemented and demonstrated |
| In-session revision + verification | Implemented and demonstrated |
| Preference eligibility gates | Implemented and demonstrated |
| DPO + LoRA training pipeline | Implemented; smoke-tested at small scale |
| HF baseline / optimized eval UI | Implemented; quantitative superiority not established |
| MCP tool interoperability | Not implemented (future work) |
| LangGraph orchestration | Not implemented (future work) |

---

## 14. Known limitations and claim boundaries

- Small local judges may emit invalid JSON; expect `agentic_degraded` on some runs.
- Preferred judge (`llama3.1:8b`) may be unavailable; demos often observe `qwen2.5:1.5b`.
- Revision quality is scenario-dependent; format constraints may remain violated after feedback.
- A single feedback cycle is not permanent learning.
- Preference gates can leave training datasets small.
- Smoke DPO (example workspace evidence: \(n=5\), `max_steps=2`, loss \(\approx 0.6938\)) is infrastructure proof only.
- No verified publication-grade baseline-versus-optimized metric table is claimed from current package evidence.
- Do not report alignment improvement percentages or statistical superiority without new controlled experiments and artifacts.
- The system is a **bounded agentic workflow**, not an unbounded autonomous agent.

---

## 15. Future research directions

Immediate:

- Stabilize supervisory structured outputs and judge model selection.
- Grow and human-spot-check preference data before non-smoke DPO.
- Export controlled HF baseline vs HF+adapter evaluations on `data/benchmark/eval_set.json`.
- Ablations: retrieval on/off, `analyze_behavior` on/off, autonomy-mode rates.

Longer term:

- Stronger supervisory models; multi-cycle feedback; latency/cost measurement.
- MCP integrations and broader tool interoperability if warranted.
- More robust orchestration (for example LangGraph) if complexity justifies it.
- Independent human evaluation; evaluator-bias and reward-hacking studies.
- QLoRA / larger bases and broader benchmarks with proper significance testing.

---

## 16. Repository layout

```
AgenticRLHF-RLAIF/
  app.py                 Streamlit application
  config.py              Central configuration
  requirements.txt
  .env.example
  src/
    pipeline/            Agentic cycle orchestration
    judge/               Agent runtime, tools, preference gates, scoring
    models/              Target / judge clients
    retrieval/           Chroma store and safety retrieval
    memory/              SQLite persistence
    training/            DPO + LoRA
    eval/                Behavioral checks and HF benchmark helpers
    demo/                Demo UI helpers
  data/
    safety/              Principles and examples (tracked JSON)
    benchmark/           Demo/eval scenario prompts (tracked JSON)
  docs/                  Version notes and design audits
  scripts/               Launch, smoke tests, retrieval init, audits
  resource material/     Technical handover package and screenshots
  checkpoints/           Local adapters (ignored; not for naive commits)
  exports/               Local run dumps (ignored)
```

---

## 17. Reproducibility and experimental integrity

- Record **configured** versus **observed** model ids for every reported run.
- Preserve autonomy mode labels (`agentic` / `agentic_degraded` / `fixed_workflow_fallback`).
- Keep interactive correction results separate from offline training claims.
- Prefer exporting machine-readable cycle results (JSON) alongside screenshots.
- Do not commit secrets, virtualenvs, Chroma binaries, SQLite databases, or raw weight files.
- Technical handover materials under `resource material/` document evidence boundaries for report writing.

---

## 18. Licence and contact

Repository: [https://github.com/abhishekreddy-msrit/AgenticRLHF-RLAIF](https://github.com/abhishekreddy-msrit/AgenticRLHF-RLAIF)

If you use this codebase in academic work, cite the repository URL and version tag or commit hash, and state clearly which components were demonstrated versus experimentally evaluated.
