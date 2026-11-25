"""Print a compact view of a target probe file (first 230 chars of each response)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.behavioral import REFUSAL_CUES, _hit  # noqa: E402

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
for row in data["results"]:
    print("=" * 100)
    print(f"[{row['family']}] {row['prompt']}")
    for i, r in enumerate(row["responses"], 1):
        head = " ".join(r.split())[:230]
        cue = _hit(REFUSAL_CUES, r)
        cue_head = _hit(REFUSAL_CUES, r[:200])
        print(f"  ({i}) len={len(r)} refusal_cue={cue} in_first_200={cue_head} :: {head}")
