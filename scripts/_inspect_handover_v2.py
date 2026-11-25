"""Final quality inspection of handover v2 DOCX + companion files."""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "resource material"
DOCX = RES / "AgenticRLHF_Application_Technical_Handover_v2.docx"
ORIG = RES / "AgenticRLHF_Application_Technical_Handover.docx"
FACTS = RES / "PROJECT_TECHNICAL_FACTS_v2.md"
NOTES = RES / "application_demo_notes_v2.md"
SHOTS = RES / "application_screenshots"
SHOTS_V2 = RES / "application_screenshots_v2"
TRAIN = ROOT / "checkpoints" / "dpo_lora" / "final_adapter" / "train_meta.json"
EXPORT = ROOT / "exports" / "handover_demo_results.json"

REQUIRED_HEADING_FRAGMENTS = [
    "Executive technical summary",
    "What the project is",
    "What is implemented",
    "What has actually been demonstrated",
    "Project objectives",
    "Application overview",
    "Architecture",
    "Live inference",
    "Offline preference",
    "Technology stack",
    "Supervisory agent",
    "Agentic feedback loop",
    "Experiment results matrix",
    "Preference data, DPO, LoRA",
    "Direct Preference Optimization",
    "LoRA",
    "Baseline vs optimized",
    "limitations",
    "Future work",
    "Claim-boundary",
    "Checklist for Pawan",
]

FORBIDDEN_CLAIM_PHRASES = [
    "significantly improves",
    "improves LLM safety by",
    "X% safer",
    "statistically significant improvement",
]


def main() -> int:
    issues: list[str] = []
    oks: list[str] = []

    for p, label in [
        (DOCX, "v2 docx"),
        (ORIG, "original docx"),
        (FACTS, "facts v2"),
        (NOTES, "notes v2"),
        (TRAIN, "train_meta"),
        (EXPORT, "demo export"),
    ]:
        if p.exists():
            oks.append(f"exists: {label} ({p.stat().st_size} bytes)")
        else:
            issues.append(f"MISSING: {label} -> {p}")

    if ORIG.exists() and DOCX.exists() and ORIG.resolve() == DOCX.resolve():
        issues.append("Original and v2 resolve to same path (overwrite risk)")
    elif ORIG.exists():
        oks.append("original path distinct from v2")

    doc = Document(str(DOCX))
    texts = []
    for p in doc.paragraphs:
        texts.append(p.text)
    full = "\n".join(texts)
    headings = [p.text.strip() for p in doc.paragraphs if p.style.name.startswith("Heading")]

    for frag in REQUIRED_HEADING_FRAGMENTS:
        if not any(frag.lower() in h.lower() for h in headings):
            # also search body in case style missed
            if frag.lower() not in full.lower():
                issues.append(f"missing section fragment: {frag}")
            else:
                oks.append(f"section present (body): {frag}")
        else:
            oks.append(f"heading present: {frag}")

    # tables
    n_tables = len(doc.tables)
    if n_tables < 8:
        issues.append(f"too few tables: {n_tables}")
    else:
        oks.append(f"tables: {n_tables}")

    # empty tables?
    for i, t in enumerate(doc.tables):
        if len(t.rows) < 2:
            issues.append(f"table {i} has <2 rows")
        # check first data cell non-empty in row1
        if t.rows and all(not c.text.strip() for c in t.rows[0].cells):
            issues.append(f"table {i} header empty")

    # images via relationships
    image_count = 0
    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            image_count += 1
    if image_count < 10:
        issues.append(f"too few embedded images: {image_count}")
    else:
        oks.append(f"embedded images: {image_count}")

    # figure captions
    fig_caps = [t for t in texts if t.startswith("Figure ")]
    oks.append(f"figure captions: {len(fig_caps)}")
    if len(fig_caps) < 10:
        issues.append(f"too few figure captions: {len(fig_caps)}")

    # claim hygiene
    low = full.lower()
    for phrase in FORBIDDEN_CLAIM_PHRASES:
        if phrase.lower() in low and "not acceptable" not in low[max(0, low.find(phrase.lower()) - 80) : low.find(phrase.lower()) + 80]:
            # allow if in the "not acceptable" example
            idx = low.find(phrase.lower())
            window = low[max(0, idx - 120) : idx + 120]
            if "not acceptable" in window or "without further evidence" in window or "example" in window:
                continue
            issues.append(f"risky claim phrase found: {phrase}")

    # verify train numbers match
    meta = json.loads(TRAIN.read_text(encoding="utf-8"))
    loss = str(meta["metrics"]["train_loss"])
    if loss[:6] not in full and "0.693" not in full and "0.694" not in full:
        issues.append("train_loss not reflected in doc")
    else:
        oks.append("train_loss reflected")

    if str(meta["n_preferences"]) not in full:
        issues.append("n_preferences missing from doc")
    else:
        oks.append("n_preferences reflected")

    # screenshots originals untouched count
    orig_shots = list(SHOTS.glob("*.png"))
    v2_shots = list(SHOTS_V2.glob("*.png"))
    oks.append(f"original screenshots: {len(orig_shots)}")
    oks.append(f"v2 prepared screenshots: {len(v2_shots)}")

    # check v2 images readable / reasonable size
    for p in v2_shots:
        im = Image.open(p)
        if im.width < 400 or im.height < 200:
            issues.append(f"v2 figure possibly too small: {p.name} {im.size}")
        else:
            oks.append(f"v2 fig ok: {p.name} {im.size}")

    # page numbers: footer field present?
    has_page = False
    for section in doc.sections:
        footer = section.footer
        xml = footer._element.xml
        if "PAGE" in xml:
            has_page = True
    if has_page:
        oks.append("page number field present")
    else:
        issues.append("page number field missing")

    # MCP/LangGraph only as future
    if "mcp" in low:
        # ensure not claimed as implemented architecture component without future/not
        for line in texts:
            ll = line.lower()
            if "mcp" in ll and "future" not in ll and "not" not in ll and "langgraph" in ll:
                if "implemented" in ll and "not" not in ll:
                    issues.append(f"possible MCP as implemented claim: {line[:120]}")

    report = {
        "ok_count": len(oks),
        "issue_count": len(issues),
        "issues": issues,
        "oks_sample": oks[:40],
        "headings": headings,
        "n_tables": n_tables,
        "n_images": image_count,
        "n_fig_caps": len(fig_caps),
        "docx_bytes": DOCX.stat().st_size,
        "orig_bytes": ORIG.stat().st_size,
    }
    out = ROOT / "exports" / "handover_v2_inspection.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
