# AgenticRLHF-RLAIF — V0.5

Canonical workspace: **`E:\Projects\AgenticRLHF-RLAIF`**

Experimental harness for agentic AI feedback used in safety alignment.

**Generate → Agent investigates (tools) → Feedback → Revise → Verify**

## What changed in V0.5

1. **Agentic Loop tab:** supervisory agent chooses tools, retrieves from Chroma, sends feedback, target revises, agent re-checks.
2. Observable action trace (decisions, tools, retrievals) — fallbacks labeled honestly.
3. Preference gates preserved; in-session correction ≠ weight update.
4. See `docs/V0.5_AGENTIC_LOOP.md`.

## What changed in V0.4

1. Rule-based score repair + preference quarantine / trusted training gates.
2. Behavioral primary eval metric. See `docs/V0.4_AUDIT.md`, `docs/V0.4_JUDGE_IMPROVEMENTS.md`.

## What changed in V0.3

1. **Storage on E:** project + `HF_HOME` + `OLLAMA_MODELS` on E:
2. **Stronger judge baseline:** two-pass + behavioral precheck; heuristic prefs blocked
3. **DPO discipline:** adapter load verification; small-N honesty warnings

## Run (from E:)

```powershell
cd E:\Projects\AgenticRLHF-RLAIF
.\scripts\run_app.ps1
```

Or:

```powershell
cd E:\Projects\AgenticRLHF-RLAIF
$env:HF_HOME="E:\ml-cache\huggingface"
$env:OLLAMA_MODELS="E:\ollama-models\models"
streamlit run app.py
```

## Tests

```powershell
cd E:\Projects\AgenticRLHF-RLAIF
$env:HF_HOME="E:\ml-cache\huggingface"
$env:OLLAMA_MODELS="E:\ollama-models\models"
python scripts\test_judge_rules.py
python scripts\benchmark_judge_v04.py
python scripts\smoke_v03.py
python scripts\smoke_demo.py
```

## Manual one-time checks

After reboot, confirm these User environment variables exist:

- `HF_HOME=E:\ml-cache\huggingface`
- `OLLAMA_MODELS=E:\ollama-models\models`

If `ollama list` is empty after reboot, start Ollama once and re-check `OLLAMA_MODELS`.

## Architecture honesty

| Role | Runtime | Model |
|---|---|---|
| Playground target | Ollama | `qwen2.5:1.5b` |
| Trainable / optimized | HF + LoRA | `SmolLM2-135M-Instruct` |
| Judge | Ollama | `llama3.1:8b` (fallback `qwen2.5:1.5b`) |
| Embeddings | Ollama | `nomic-embed-text` |

HF LoRA adapters **cannot** load into Ollama GGUF models.

## Faculty Demo Mode

Use the **Demo Mode** tab for the reproducible loop. Reset clears only `data/demo_rlaif.db`.

## Legacy location

`D:\RIT\Third Year\Prompt Engineering\AgenticRLHF-RLAIF` is a pre-migration copy and should be treated as obsolete once E: is confirmed working.
