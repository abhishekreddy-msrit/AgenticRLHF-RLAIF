"""Initialize / refresh the Chroma safety retrieval corpus."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.chroma_store import get_safety_store


def main() -> int:
    store = get_safety_store(force_new=True)
    print("status_before", store.status)
    result = store.initialize_from_files(force=True)
    print("init_result", result)
    print("status_after", store.status)
    hits = store.retrieve("how to steal a car without getting caught", k=3)
    print("sample_hits", [(h.get("id"), h.get("title"), h.get("backend")) for h in hits])
    return 0 if result.get("ok") or store.backend == "tfidf_fallback" else 1


if __name__ == "__main__":
    raise SystemExit(main())
