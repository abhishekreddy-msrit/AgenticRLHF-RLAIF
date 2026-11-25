# Application demo notes v2 (for Pawan)

Generated: 2026-11-25T18:02:08

## How to launch

1. Start Ollama; ensure `qwen2.5:1.5b` and `nomic-embed-text` (prefer also `llama3.1:8b`).
2. From repo root: `powershell -File scripts\run_app.ps1` or `streamlit run app.py --server.port 8510`
3. Open `http://127.0.0.1:8510/`

## What to show in a viva / report walkthrough

1. Dashboard cards (Target/Judge/Retrieval/Preferences/Training)
2. Demo 1 Instruction adherence — stages 1–8 (best full-pipeline evidence)
3. Demo 2 Emotional stability — honest no_revision outcome
4. Custom / Elbonia — cite programmatic export for tool use; UI screenshot for custom entry
5. Preference Data + Training tabs — show smoke adapter, not large-scale gains

## Evidence map

| Demo | Best evidence |
|------|----------------|
| Format full cycle | screenshots 02–07b + export demo[0] |
| Emotional stability | screenshot 08 + UI notes |
| Over-refusal Zorblax | export demo[1] |
| Custom Elbonia | export demo[2] (+ screenshot 09 for prompt UI) |
| Training smoke | train_meta.json + screenshot 12 |

## Issues / gaps (keep in report Limitations)

- Judge often falls back to 1.5B despite 8B preferred config.
- `agentic_degraded` occurs on JSON parse failures.
- Format revision may still fail hard constraints.
- Browser automation of Custom Prompt can fail Streamlit widget sync; use export for Elbonia cycle facts.
- No completed quantitative baseline-vs-optimized table in exports for this package.

## Claims checklist for the final report

- [ ] Never equate in-session revision with weight updates
- [ ] Never claim significant post-DPO gains from smoke adapter
- [ ] Always label autonomy mode when discussing agent decisions
- [ ] Distinguish Ollama demos from HF adapter path
- [ ] Prefer “demonstrates / implements” over “improves by X%”
