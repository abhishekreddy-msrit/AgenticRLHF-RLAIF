"""Lightweight local retrieval over safety principles (no external vector DB required)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config


class SafetyRetriever:
    def __init__(self, principles_path: Path | None = None):
        path = Path(principles_path or (config.SAFETY_DIR / "principles.json"))
        with path.open(encoding="utf-8") as f:
            self.principles: list[dict] = json.load(f)
        self.docs = [
            " ".join(
                [
                    p.get("title", ""),
                    p.get("category", ""),
                    p.get("principle", ""),
                    p.get("example_bad", ""),
                    p.get("example_good", ""),
                ]
            )
            for p in self.principles
        ]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.docs)

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        if not query.strip():
            return self.principles[:k]
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        idx = np.argsort(-sims)[:k]
        out = []
        for i in idx:
            item = dict(self.principles[int(i)])
            item["score"] = float(sims[int(i)])
            out.append(item)
        return out

    def retrieve_for_prompt_response(self, prompt: str, response: str, k: int = 3) -> list[dict]:
        query = f"{prompt}\n{response}"
        # Boost categories by simple keyword cues
        cues = {
            "harmful_intent": r"\b(kill|explosive|steal|hack|weapon|poison|bomb)\b",
            "manipulation": r"\b(ignore previous|jailbreak|DAN|developer mode|uncensored)\b",
            "emotional_escalation": r"\b(hate|worthless|kill myself|angry|stupid)\b",
            "truthfulness": r"\b(exact|cite|doi|official report|100% sure)\b",
            "system_adherence": r"\b(system prompt|hidden rules|instructions verbatim)\b",
            "over_refusal": r"\b(recipe|study|python|coding|kitchen)\b",
        }
        boosted = list(self.retrieve(query, k=k))
        for cat, pattern in cues.items():
            if re.search(pattern, query, flags=re.I):
                for p in self.principles:
                    if p["category"] == cat and p["id"] not in {b["id"] for b in boosted}:
                        item = dict(p)
                        item["score"] = 0.99
                        boosted.insert(0, item)
                        break
        return boosted[:k]
