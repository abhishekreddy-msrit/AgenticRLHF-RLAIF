"""Persistent ChromaDB retrieval with Ollama embeddings and TF-IDF fallback."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import config
from src.models.ollama_client import OllamaClient
from src.retrieval.safety_retriever import SafetyRetriever


class _OllamaEmbeddingFunction:
    """Chroma-compatible embedding function using Ollama nomic-embed-text."""

    def __init__(self, model: str | None = None, client: OllamaClient | None = None):
        self.model = model or config.OLLAMA_EMBED_MODEL
        self.client = client or OllamaClient()

    def name(self) -> str:
        return f"ollama-{self.model}"

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return [self.client.embed(self.model, text) for text in input]

    def embed_query(self, input: list[str]) -> list[list[float]]:
        # Chroma may pass a list or a single string depending on version.
        if isinstance(input, str):
            return [self.client.embed(self.model, input)]
        return [self.client.embed(self.model, text) for text in input]

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embed_documents(input)


class ChromaSafetyStore:
    """
    Persistent local semantic store for judge context.
    Falls back to TF-IDF if Chroma or embeddings are unavailable.
    """

    COLLECTION = "safety_alignment"

    def __init__(self, persist_dir: Path | None = None):
        self.persist_dir = Path(persist_dir or config.CHROMA_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.backend = "unavailable"
        self.fallback_reason: str | None = None
        self._collection = None
        self._tfidf = SafetyRetriever()
        self._client = OllamaClient()
        self._init_chroma()

    def _init_chroma(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings

            if not self._client.has_model(config.OLLAMA_EMBED_MODEL):
                raise RuntimeError(
                    f"Embedding model '{config.OLLAMA_EMBED_MODEL}' not found in Ollama"
                )

            emb = _OllamaEmbeddingFunction(client=self._client)
            # Probe one embedding to fail fast
            _ = emb(["probe safety retrieval"])

            client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = client.get_or_create_collection(
                name=self.COLLECTION,
                embedding_function=emb,
                metadata={"hnsw:space": "cosine"},
            )
            self.backend = "chroma"
            self.fallback_reason = None
        except Exception as e:
            self._collection = None
            self.backend = "tfidf_fallback"
            self.fallback_reason = str(e)

    @property
    def status(self) -> dict[str, Any]:
        count = 0
        if self._collection is not None:
            try:
                count = int(self._collection.count())
            except Exception:
                count = 0
        return {
            "backend": self.backend,
            "fallback_reason": self.fallback_reason,
            "persist_dir": str(self.persist_dir),
            "document_count": count,
            "embed_model": config.OLLAMA_EMBED_MODEL if self.backend == "chroma" else None,
        }

    @staticmethod
    def _doc_id(prefix: str, raw_id: str) -> str:
        digest = hashlib.sha1(f"{prefix}:{raw_id}".encode("utf-8")).hexdigest()[:16]
        return f"{prefix}-{digest}"

    def upsert_documents(self, docs: list[dict[str, Any]]) -> int:
        """Insert/update docs. Uses stable IDs to avoid duplicates."""
        if self._collection is None:
            return 0
        ids, documents, metadatas = [], [], []
        for d in docs:
            did = str(d["id"])
            text = str(d["text"]).strip()
            if not text:
                continue
            ids.append(did)
            documents.append(text)
            metadatas.append(
                {
                    "source_id": str(d.get("source_id") or did),
                    "category": str(d.get("category") or "general"),
                    "title": str(d.get("title") or ""),
                    "doc_type": str(d.get("doc_type") or "principle"),
                }
            )
        if not ids:
            return 0
        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(ids)

    def initialize_from_files(self, force: bool = False) -> dict[str, Any]:
        """Populate collection from local safety JSON corpora."""
        principles_path = config.SAFETY_DIR / "principles.json"
        examples_path = config.SAFETY_DIR / "examples.json"
        principles = json.loads(principles_path.read_text(encoding="utf-8"))
        examples = []
        if examples_path.exists():
            examples = json.loads(examples_path.read_text(encoding="utf-8"))

        docs: list[dict[str, Any]] = []
        for p in principles:
            text = (
                f"{p.get('title','')}. {p.get('principle','')} "
                f"Bad example: {p.get('example_bad','')} "
                f"Good example: {p.get('example_good','')}"
            )
            docs.append(
                {
                    "id": self._doc_id("principle", p["id"]),
                    "source_id": p["id"],
                    "category": p.get("category", "principle"),
                    "title": p.get("title", ""),
                    "doc_type": "principle",
                    "text": text,
                }
            )
        for ex in examples:
            docs.append(
                {
                    "id": self._doc_id("example", ex["id"]),
                    "source_id": ex["id"],
                    "category": ex.get("category", "example"),
                    "title": ex.get("title", ""),
                    "doc_type": "example",
                    "text": f"{ex.get('title','')}. {ex.get('text','')}",
                }
            )

        if self.backend != "chroma":
            return {
                "ok": False,
                "backend": self.backend,
                "added": 0,
                "reason": self.fallback_reason,
                "prepared_docs": len(docs),
            }

        existing = int(self._collection.count()) if self._collection else 0
        if existing > 0 and not force:
            return {
                "ok": True,
                "backend": self.backend,
                "added": 0,
                "skipped": True,
                "document_count": existing,
                "message": "Collection already populated (use force=True to re-upsert).",
            }

        n = self.upsert_documents(docs)
        return {
            "ok": True,
            "backend": self.backend,
            "added": n,
            "document_count": int(self._collection.count()) if self._collection else n,
        }

    def add_evaluation_case(self, evaluation: dict[str, Any]) -> str | None:
        if self.backend != "chroma":
            return None
        eid = str(evaluation.get("id") or hashlib.sha1(repr(evaluation).encode()).hexdigest()[:12])
        text = (
            f"Prompt: {evaluation.get('prompt','')}\n"
            f"Response: {evaluation.get('target_response','')}\n"
            f"Assessment: {evaluation.get('overall_assessment','')}\n"
            f"Violations: {evaluation.get('violations', [])}\n"
            f"Critique: {evaluation.get('critique','')}"
        )
        doc_id = self._doc_id("eval", eid)
        self.upsert_documents(
            [
                {
                    "id": doc_id,
                    "source_id": eid,
                    "category": "prior_evaluation",
                    "title": f"Evaluation {eid[:8]}",
                    "doc_type": "evaluation",
                    "text": text[:2000],
                }
            ]
        )
        return doc_id

    def add_preference_example(self, pair: dict[str, Any]) -> str | None:
        if self.backend != "chroma":
            return None
        pid = str(pair.get("id") or hashlib.sha1(repr(pair).encode()).hexdigest()[:12])
        text = (
            f"Prompt: {pair.get('prompt','')}\n"
            f"Chosen: {pair.get('chosen','')}\n"
            f"Rejected: {pair.get('rejected','')}"
        )
        doc_id = self._doc_id("pref", pid)
        self.upsert_documents(
            [
                {
                    "id": doc_id,
                    "source_id": pid,
                    "category": pair.get("category") or "preference",
                    "title": f"Preference {pid[:8]}",
                    "doc_type": "preference",
                    "text": text[:2000],
                }
            ]
        )
        return doc_id

    def retrieve(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        k = k or config.RETRIEVAL_TOP_K
        if self.backend == "chroma" and self._collection is not None:
            try:
                res = self._collection.query(query_texts=[query], n_results=k)
                out = []
                ids = (res.get("ids") or [[]])[0]
                docs = (res.get("documents") or [[]])[0]
                metas = (res.get("metadatas") or [[]])[0]
                dists = (res.get("distances") or [[]])[0]
                for i, doc_id in enumerate(ids):
                    meta = metas[i] if i < len(metas) else {}
                    dist = dists[i] if i < len(dists) else None
                    score = None if dist is None else round(1.0 / (1.0 + float(dist)), 4)
                    out.append(
                        {
                            "id": meta.get("source_id") or doc_id,
                            "chroma_id": doc_id,
                            "title": meta.get("title") or "",
                            "category": meta.get("category") or "",
                            "doc_type": meta.get("doc_type") or "",
                            "principle": docs[i] if i < len(docs) else "",
                            "text": docs[i] if i < len(docs) else "",
                            "score": score,
                            "backend": "chroma",
                        }
                    )
                return out
            except Exception as e:
                self.fallback_reason = f"Chroma query failed: {e}"
                self.backend = "tfidf_fallback"

        # TF-IDF fallback (still useful; clearly labeled)
        hits = self._tfidf.retrieve(query, k=k)
        out = []
        for h in hits:
            out.append(
                {
                    "id": h.get("id"),
                    "title": h.get("title"),
                    "category": h.get("category"),
                    "doc_type": "principle",
                    "principle": h.get("principle"),
                    "text": h.get("principle"),
                    "example_good": h.get("example_good"),
                    "example_bad": h.get("example_bad"),
                    "score": h.get("score"),
                    "backend": "tfidf_fallback",
                }
            )
        return out

    def retrieve_for_prompt_response(
        self, prompt: str, response: str, k: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Prefer principles/examples over prior evaluations/preferences so the judge
        is grounded in behavioral policy rather than contaminated training echoes.
        """
        k = k or config.RETRIEVAL_TOP_K
        # Over-fetch then re-rank
        hits = self.retrieve(f"{prompt}\n{response}", k=max(k * 3, k))
        preferred_types = {"principle", "example", ""}
        principles = [
            h
            for h in hits
            if (h.get("doc_type") in preferred_types)
            or str(h.get("id") or "").startswith("safe-")
            or str(h.get("id") or "").startswith("ex-")
        ]
        other = [h for h in hits if h not in principles]
        ranked = principles + other
        # Deduplicate by id preserving order
        seen = set()
        out = []
        for h in ranked:
            hid = h.get("id") or h.get("chroma_id")
            if hid in seen:
                continue
            seen.add(hid)
            out.append(h)
            if len(out) >= k:
                break
        return out


_STORE: ChromaSafetyStore | None = None


def get_safety_store(force_new: bool = False) -> ChromaSafetyStore:
    global _STORE
    if _STORE is None or force_new:
        _STORE = ChromaSafetyStore()
        if _STORE.backend == "chroma":
            _STORE.initialize_from_files(force=False)
    return _STORE
