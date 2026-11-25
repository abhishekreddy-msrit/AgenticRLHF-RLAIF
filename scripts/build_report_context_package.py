"""Build AgenticRLHF_Report_Context_Package.zip for Pavan/Claude report drafting.

Does not modify originals. Excludes secrets, weights, caches, and obsolete duplicates.
"""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "resource material"
STAGE = RES / "_staging_report_context_package"
ZIP_PATH = RES / "AgenticRLHF_Report_Context_Package.zip"
PKG_ROOT_NAME = "AgenticRLHF_Report_Context_Package"

# Flag only literal secret *values*, not env-var name references in source.
SECRET_VALUE_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|password|access[_-]?token)\s*[=:]\s*['\"][^'\"]{12,}['\"]"),
    re.compile(r"(?i)\bsk-[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)Bearer\s+[A-Za-z0-9\-._~+/]{20,}={0,2}"),
]


def copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def scan_secrets(path: Path) -> list[str]:
    hits: list[str] = []
    if path.suffix.lower() not in {
        ".py",
        ".md",
        ".txt",
        ".json",
        ".yml",
        ".yaml",
        ".env",
        ".example",
        ".toml",
        ".cfg",
        ".ini",
    }:
        return hits
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return hits
    for pat in SECRET_VALUE_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        snippet = m.group(0)[:80]
        # Ignore code that only references environment variable names / placeholders
        if any(
            x in snippet
            for x in (
                "os.environ",
                "os.getenv",
                "OPENAI_API_KEY",
                "GROQ_API_KEY",
                "GOOGLE_API_KEY",
                "GEMINI_API_KEY",
                "your_",
                "CHANGEME",
                "placeholder",
            )
        ):
            continue
        if "Bearer {api_key}" in text[m.start() : m.end() + 5] or "Bearer {api_key}" in snippet:
            continue
        hits.append(f"{path.name}: literal-looking secret pattern near {snippet!r}")
    return hits


def write_source_index(pkg: Path) -> None:
    text = """# SOURCE_CODE_INDEX.md — Curated implementation reference

Purpose: Help Claude/Pavan locate the code that implements each subsystem described in the technical handover.
Paths below are relative to `source_code_reference/` and mirror the repository layout.

## Entry points and configuration

| File | Why it matters for the report |
|------|-------------------------------|
| `app.py` | Streamlit UI: Agentic Loop, Demo Mode, Preference Data, Training, Evaluation tabs |
| `config.py` | Central models, paths, judge thresholds, training defaults (APP_VERSION 0.5) |
| `requirements.txt` | Python dependency stack (Streamlit, Transformers, PEFT, TRL, Chroma, etc.) |
| `.env.example` | Safe shareable configuration template (no secrets). Do **not** treat as live `.env` |
| `README.md` | Project overview and dual-path model honesty table |

## Live agentic feedback loop

| File | Role |
|------|------|
| `src/pipeline/agentic_feedback.py` | Orchestrates investigate → feedback → revise → verify → preference gate |
| `src/pipeline/cycle_schema.py` | Structured cycle / stage data shapes used by the UI |
| `src/pipeline/loop.py` | Related pipeline loop helpers |
| `src/demo/agentic_panel.py` | Streamlit staging UI for the agentic cycle |
| `src/demo/prepare.py` | Demo preparation helpers |

## Supervisory agent / judge / tools

| File | Role |
|------|------|
| `src/judge/agentic_runtime.py` | ReAct-style autonomy: agentic / agentic_degraded / fixed_workflow_fallback |
| `src/judge/agent.py` | Agent decision / investigation logic |
| `src/judge/tool_registry.py` | Tool registration for the supervisory agent |
| `src/judge/tools.py` | Tool implementations (retrieve, analyze, inspect, etc.) |
| `src/judge/preference.py` | Preference pair creation and eligibility/trust gates |
| `src/judge/verifier.py` | Rule / deterministic verification helpers |
| `src/judge/scoring.py` | Score assembly / repair related logic |
| `src/judge/prompts.py` | Judge/agent prompts |
| `src/judge/schemas.py` | Structured judgment schemas |
| `src/judge/rubric.py` | Evaluation rubric |

## Models

| File | Role |
|------|------|
| `src/models/target_llm.py` | Target LLM interface (Ollama / HF dual path) |
| `src/models/judge_llm.py` | Judge / supervisory LLM client configuration |
| `src/models/ollama_client.py` | Ollama HTTP client |

## Retrieval and memory

| File | Role |
|------|------|
| `src/retrieval/chroma_store.py` | Chroma vector store setup and queries |
| `src/retrieval/safety_retriever.py` | Safety principle / example retrieval |
| `src/memory/db.py` | SQLite persistence (conversations, evaluations, preferences) |

## Offline DPO + LoRA and evaluation

| File | Role |
|------|------|
| `src/training/dpo_train.py` | TRL DPOTrainer + PEFT LoRA training entry |
| `src/training/feasibility.py` | Training feasibility / honesty checks |
| `src/eval/benchmark.py` | HF baseline vs HF+adapter evaluation helpers |
| `src/eval/behavioral.py` | Behavioral evaluation logic |
| `src/eval/behavioral_metrics.py` | Behavioral metric helpers |
| `src/eval/constraints.py` | Format / constraint checking used in verification |

## Reading tip for code-backed report sections

1. Start with `app.py` + `src/pipeline/agentic_feedback.py` for the live loop narrative.
2. Read `src/judge/agentic_runtime.py` + `tool_registry.py` + `tools.py` for agent autonomy and tools.
3. Read `preference.py` then `dpo_train.py` for preference → offline training separation.
4. Read `benchmark.py` only as **infrastructure** for baseline/optimized eval — do not invent scores.
"""
    (pkg / "SOURCE_CODE_INDEX.md").write_text(text, encoding="utf-8")


def write_readme(pkg: Path) -> None:
    text = f"""# AgenticRLHF Report Context Package

Generated: {datetime.now().isoformat(timespec="seconds")}  
Prepared for: **Pavan** (final academic project report drafting with Claude)  
Project: AgenticRLHF-RLAIF (Application V0.5)

## Purpose

This ZIP is a **curated, evidence-bounded context package** for writing the final academic report.
It is **not** the final report itself.

Use it so Claude has the technical architecture, demonstration evidence, claim boundaries, and
relevant source code without needing the entire repository, model weights, or local caches.

## Package contents

```
AgenticRLHF_Report_Context_Package/
├── README_CONTEXT_PACKAGE.md          ← this file
├── SOURCE_CODE_INDEX.md               ← curated code map
├── technical_documentation/           ← authoritative technical sources (v2)
├── application_screenshots/           ← report-ready figures (v2)
├── source_code_reference/             ← curated implementation files
├── evidence_artifacts/                ← verified demo + training metadata
├── methodology_data/                  ← scenarios, eval prompts, principles
└── project_background/                ← README + version/design notes
```

### technical_documentation/ (authoritative)

| File | Authority |
|------|-----------|
| `AgenticRLHF_Application_Technical_Handover_v2.docx` | **Primary** application + technical handover |
| `PROJECT_TECHNICAL_FACTS_v2.md` | **Primary** verified facts + claim boundaries |
| `application_demo_notes_v2.md` | Demo instructions, observed outcomes, gaps |

### application_screenshots/

Cropped/enlarged genuine UI screenshots suitable for report figures.
Filenames preserved from `application_screenshots_v2/`.

Strong figure candidates:

- `00_architecture_diagram.png` — live loop vs offline DPO+LoRA
- `01_application_dashboard.png` — main UI
- `02`–`07b_demo1_*.png` — full instruction-adherence pipeline evidence
- `08_demo2_full_cycle.png` — emotional stability / no_revision_needed
- `09_demo3_custom_prompt.png` — custom prompt UI
- `12_training_tab.png`, `13_preference_data_tab.png` — training / preferences UI

### source_code_reference/

Curated implementation files with original relative paths. See `SOURCE_CODE_INDEX.md`.

### evidence_artifacts/

| File | Use |
|------|-----|
| `exports/handover_demo_results.json` | Authoritative programmatic demo outcomes |
| `exports/handover_v2_evidence_pack.json` | Consolidated evidence pack used for handover v2 |
| `checkpoints/.../train_meta.json` | Smoke DPO+LoRA metrics (n=5, max_steps=2, loss≈0.6938) |
| `checkpoints/.../adapter_config.json` | LoRA config (r=8, alpha=16) — **no weight files** |

### methodology_data/

Scenario prompts, held-out eval set prompts, and safety principles/examples used by retrieval.
Useful for Methodology sections — not fabricated results.

### project_background/

| File | Notes |
|------|-------|
| `README.md` | Project overview |
| `docs/V0.5_AGENTIC_LOOP.md` | V0.5 agentic loop design notes |
| `docs/V0.4_JUDGE_IMPROVEMENTS.md` | Judge/rule improvements |
| `docs/V0.4_AUDIT.md` | V0.4 audit notes |
| `docs/V0.3_ASSESSMENT.md` | Earlier assessment notes |

**Not found in the accessible project material (do not invent):**

- Institutional report template
- Formal project proposal / synopsis / literature-survey Word docs

If Pavan has those elsewhere (college LMS / shared drive), add them separately.

## Recommended reading order for Claude

1. `technical_documentation/PROJECT_TECHNICAL_FACTS_v2.md`
2. `technical_documentation/AgenticRLHF_Application_Technical_Handover_v2.docx`
3. `technical_documentation/application_demo_notes_v2.md`
4. `SOURCE_CODE_INDEX.md`
5. Relevant files under `source_code_reference/`
6. `application_screenshots/` (+ captions from the handover)
7. `evidence_artifacts/` for exact demo/training numbers
8. `project_background/` and `methodology_data/` as needed

## Hard claim boundaries (do not soften)

- The **live agentic feedback loop is implemented and demonstrated**.
- The **live loop does not directly update model weights**.
- The **DPO + LoRA pipeline exists** but was **smoke-tested at very small scale**.
- Smoke test used **5 preferences** and **2 training steps**.
- Reported smoke-training **loss ≈ 0.6938**.
- **Adapter loading was verified**.
- **No verified quantitative baseline-versus-optimized performance improvement** is established by available evidence.
- Do **not** claim alignment improvement percentages or statistical superiority.
- Do **not** present **MCP** or **LangGraph** as implemented features (future work only).
- Distinguish **configured** models (e.g. preferred judge `llama3.1:8b`) from **observed** demo models (often `qwen2.5:1.5b`).
- Demonstrated outcomes include **revision**, **unchanged** verification, **no_revision_needed**, and **agentic_degraded** execution.

## Evidence vs implementation summary

| Topic | Status |
|-------|--------|
| Target + supervisory agent + tools + retrieval + revision + verify | Implemented + Demonstrated |
| Preference gates + SQLite storage | Implemented + Demonstrated |
| DPO+LoRA training path | Implemented + Tested (smoke) |
| Meaningful post-DPO alignment gains | Not yet established |
| Baseline vs optimized metric superiority | Not verified from available evidence |
| MCP / LangGraph | Future work only |

## What was intentionally excluded

- Entire git history / `.git/`
- Virtual environments, `__pycache__`, pytest caches
- Model weight files (`adapter_model.safetensors`, HF/Ollama caches)
- SQLite databases and Chroma binary indexes
- Live `.env` (secrets risk) — only `.env.example` included
- Obsolete v1 handover / facts / demo notes / uncropped screenshot set (superseded by v2)
- Large unrelated exports and temporary scripts

## How to use with Claude

Provide this ZIP (or extracted folder) and ask Claude to draft the academic report **only from these materials**, respecting the claim boundaries above, and using the technical handover as the primary architecture/results source.
"""
    (pkg / "README_CONTEXT_PACKAGE.md").write_text(text, encoding="utf-8")


def stage_package() -> Path:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    pkg = STAGE / PKG_ROOT_NAME
    pkg.mkdir(parents=True)

    # A. Technical docs (v2 only)
    tech = pkg / "technical_documentation"
    for name in [
        "AgenticRLHF_Application_Technical_Handover_v2.docx",
        "PROJECT_TECHNICAL_FACTS_v2.md",
        "application_demo_notes_v2.md",
    ]:
        copy_file(RES / name, tech / name)

    # B. Screenshots v2 (no recompress)
    shots_src = RES / "application_screenshots_v2"
    shots_dst = pkg / "application_screenshots"
    for p in sorted(shots_src.glob("*.png")):
        copy_file(p, shots_dst / p.name)

    # C. Source code reference
    code_files = [
        "app.py",
        "config.py",
        "requirements.txt",
        ".env.example",
        "README.md",
        "src/pipeline/agentic_feedback.py",
        "src/pipeline/cycle_schema.py",
        "src/pipeline/loop.py",
        "src/demo/agentic_panel.py",
        "src/demo/prepare.py",
        "src/judge/agentic_runtime.py",
        "src/judge/agent.py",
        "src/judge/tool_registry.py",
        "src/judge/tools.py",
        "src/judge/preference.py",
        "src/judge/verifier.py",
        "src/judge/scoring.py",
        "src/judge/prompts.py",
        "src/judge/schemas.py",
        "src/judge/rubric.py",
        "src/models/target_llm.py",
        "src/models/judge_llm.py",
        "src/models/ollama_client.py",
        "src/retrieval/chroma_store.py",
        "src/retrieval/safety_retriever.py",
        "src/memory/db.py",
        "src/training/dpo_train.py",
        "src/training/feasibility.py",
        "src/eval/benchmark.py",
        "src/eval/behavioral.py",
        "src/eval/behavioral_metrics.py",
        "src/eval/constraints.py",
    ]
    code_root = pkg / "source_code_reference"
    for rel in code_files:
        src = ROOT / rel
        if not src.exists():
            raise FileNotFoundError(rel)
        copy_file(src, code_root / rel)

    # D. Evidence artifacts (no weights)
    evid = pkg / "evidence_artifacts"
    evidence_files = [
        ("exports/handover_demo_results.json", "exports/handover_demo_results.json"),
        ("exports/handover_v2_evidence_pack.json", "exports/handover_v2_evidence_pack.json"),
        (
            "checkpoints/dpo_lora/final_adapter/train_meta.json",
            "checkpoints/dpo_lora/final_adapter/train_meta.json",
        ),
        (
            "checkpoints/dpo_lora/final_adapter/adapter_config.json",
            "checkpoints/dpo_lora/final_adapter/adapter_config.json",
        ),
    ]
    for src_rel, dst_rel in evidence_files:
        copy_file(ROOT / src_rel, evid / dst_rel)

    # E. Methodology data
    meth = pkg / "methodology_data"
    for rel in [
        "data/benchmark/eval_set.json",
        "data/benchmark/agentic_demo_scenarios.json",
        "data/benchmark/demo_scenarios.json",
        "data/safety/principles.json",
        "data/safety/examples.json",
    ]:
        copy_file(ROOT / rel, meth / rel)

    # F. Project background (no fabricated proposal)
    bg = pkg / "project_background"
    copy_file(ROOT / "README.md", bg / "README.md")
    for name in [
        "V0.5_AGENTIC_LOOP.md",
        "V0.4_JUDGE_IMPROVEMENTS.md",
        "V0.4_AUDIT.md",
        "V0.3_ASSESSMENT.md",
    ]:
        copy_file(ROOT / "docs" / name, bg / "docs" / name)

    write_readme(pkg)
    write_source_index(pkg)

    # NOTE about missing institutional materials
    (bg / "MISSING_MATERIALS_NOTE.md").write_text(
        "No institutional report template, formal proposal, synopsis, or literature-survey "
        "document was found inside the accessible AgenticRLHF-RLAIF project tree at packaging "
        "time. If those exist outside this repository, add them manually before sharing.\n",
        encoding="utf-8",
    )
    return pkg


def zip_package(pkg: Path) -> Path:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(pkg.rglob("*")):
            if path.is_file():
                arc = path.relative_to(STAGE).as_posix()
                zf.write(path, arcname=arc)
    return ZIP_PATH


def verify(pkg: Path, zpath: Path) -> dict:
    issues: list[str] = []
    oks: list[str] = []

    must = [
        pkg / "README_CONTEXT_PACKAGE.md",
        pkg / "SOURCE_CODE_INDEX.md",
        pkg / "technical_documentation" / "AgenticRLHF_Application_Technical_Handover_v2.docx",
        pkg / "technical_documentation" / "PROJECT_TECHNICAL_FACTS_v2.md",
        pkg / "technical_documentation" / "application_demo_notes_v2.md",
        pkg / "application_screenshots" / "00_architecture_diagram.png",
        pkg / "source_code_reference" / "app.py",
        pkg / "source_code_reference" / "src" / "pipeline" / "agentic_feedback.py",
        pkg / "source_code_reference" / "src" / "training" / "dpo_train.py",
        pkg / "evidence_artifacts" / "exports" / "handover_demo_results.json",
        pkg / "evidence_artifacts" / "checkpoints" / "dpo_lora" / "final_adapter" / "train_meta.json",
    ]
    for p in must:
        if p.exists():
            oks.append(f"present: {p.relative_to(pkg)}")
        else:
            issues.append(f"missing: {p}")

    # no secrets / weights / env
    forbidden_names = {".env", "adapter_model.safetensors"}
    secret_hits: list[str] = []
    for f in pkg.rglob("*"):
        if not f.is_file():
            continue
        if f.name in forbidden_names:
            issues.append(f"forbidden packaged: {f}")
        if f.suffix.lower() == ".safetensors":
            issues.append(f"weight file packaged: {f}")
        secret_hits.extend(scan_secrets(f))
    # filter MAX_NEW_TOKENS false positives already handled; keep remaining
    for h in secret_hits:
        if "MAX_NEW_TOKENS" in h or "JUDGE_MAX_TOKENS" in h:
            continue
        issues.append(f"secret-like: {h}")

    # zip openable
    with zipfile.ZipFile(zpath, "r") as zf:
        names = zf.namelist()
        bad = zf.testzip()
        if bad:
            issues.append(f"corrupt zip member: {bad}")
        else:
            oks.append(f"zip readable members={len(names)}")
        for need in [
            f"{PKG_ROOT_NAME}/README_CONTEXT_PACKAGE.md",
            f"{PKG_ROOT_NAME}/technical_documentation/AgenticRLHF_Application_Technical_Handover_v2.docx",
            f"{PKG_ROOT_NAME}/technical_documentation/PROJECT_TECHNICAL_FACTS_v2.md",
            f"{PKG_ROOT_NAME}/application_screenshots/01_application_dashboard.png",
            f"{PKG_ROOT_NAME}/source_code_reference/app.py",
        ]:
            if need not in names:
                issues.append(f"zip missing: {need}")

        # ensure no .env or weights inside zip
        for n in names:
            base = Path(n).name
            if base == ".env" or n.endswith(".safetensors"):
                issues.append(f"zip contains forbidden: {n}")

    size_mb = zpath.stat().st_size / (1024 * 1024)
    if size_mb > 80:
        issues.append(f"zip unusually large: {size_mb:.1f} MB")
    else:
        oks.append(f"zip size_mb={size_mb:.2f}")

    # originals intact
    for name in [
        "AgenticRLHF_Application_Technical_Handover_v2.docx",
        "PROJECT_TECHNICAL_FACTS_v2.md",
        "application_demo_notes_v2.md",
        "AgenticRLHF_Application_Technical_Handover.docx",
    ]:
        if not (RES / name).exists():
            issues.append(f"original resource missing after packaging: {name}")

    report = {
        "zip": str(zpath),
        "size_bytes": zpath.stat().st_size,
        "size_mb": round(size_mb, 2),
        "issue_count": len(issues),
        "issues": issues,
        "ok_count": len(oks),
        "oks": oks,
        "staged_file_count": sum(1 for _ in pkg.rglob("*") if _.is_file()),
        "screenshot_count": len(list((pkg / "application_screenshots").glob("*.png"))),
        "source_file_count": sum(1 for _ in (pkg / "source_code_reference").rglob("*") if _.is_file()),
    }
    out = ROOT / "exports" / "report_context_package_verification.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    pkg = stage_package()
    zpath = zip_package(pkg)
    report = verify(pkg, zpath)
    # cleanup staging to avoid clutter; ZIP is the deliverable
    shutil.rmtree(STAGE, ignore_errors=True)
    print(json.dumps(report, indent=2))
    return 1 if report["issue_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
