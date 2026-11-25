"""SQLite persistence for evaluations, preferences, and experiments."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import config


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_conn(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    conn = _connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | None = None) -> None:
    with get_conn(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                target_response TEXT NOT NULL,
                result_json TEXT NOT NULL,
                overall_score REAL,
                confidence REAL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS preferences (
                id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                chosen TEXT NOT NULL,
                rejected TEXT NOT NULL,
                judge_result_id TEXT,
                scores_json TEXT,
                overall_score REAL,
                confidence REAL,
                category TEXT,
                source TEXT,
                meta_json TEXT,
                created_at TEXT NOT NULL,
                training_status TEXT DEFAULT 'legacy_unreviewed',
                quarantine_reason TEXT
            );

            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT,
                method TEXT,
                config_json TEXT,
                status TEXT,
                metrics_json TEXT,
                checkpoint_path TEXT,
                log_text TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        # Backward-compatible migration for older DBs
        cols = {r[1] for r in conn.execute("PRAGMA table_info(preferences)").fetchall()}
        if "meta_json" not in cols:
            conn.execute("ALTER TABLE preferences ADD COLUMN meta_json TEXT")
        if "training_status" not in cols:
            conn.execute(
                "ALTER TABLE preferences ADD COLUMN training_status TEXT DEFAULT 'legacy_unreviewed'"
            )
        if "quarantine_reason" not in cols:
            conn.execute("ALTER TABLE preferences ADD COLUMN quarantine_reason TEXT")


class MemoryStore:
    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path or config.DB_PATH)
        init_db(self.db_path)

    def save_evaluation(self, result: dict[str, Any]) -> str:
        with get_conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO evaluations
                (id, prompt, target_response, result_json, overall_score, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result["id"],
                    result["prompt"],
                    result["target_response"],
                    json.dumps(result),
                    result.get("overall_score"),
                    result.get("confidence"),
                    result.get("created_at") or datetime.now(timezone.utc).isoformat(),
                ),
            )
        return result["id"]

    def save_preference(self, pair: dict[str, Any]) -> str:
        meta = {
            "rationale": pair.get("rationale"),
            "score_explanation": pair.get("score_explanation"),
            "retrieved_principle_ids": pair.get("retrieved_principle_ids") or [],
            "retrieved_references": pair.get("retrieved_references") or [],
            "judgment_mode": pair.get("judgment_mode"),
            "judge_model": pair.get("judge_model"),
            "target_model": pair.get("target_model"),
            "target_backend": pair.get("target_backend"),
            "training_status": pair.get("training_status"),
            "quarantine_reason": pair.get("quarantine_reason"),
        }
        status = pair.get("training_status") or "candidate"
        with get_conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO preferences
                (id, prompt, chosen, rejected, judge_result_id, scores_json,
                 overall_score, confidence, category, source, meta_json, created_at,
                 training_status, quarantine_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pair["id"],
                    pair["prompt"],
                    pair["chosen"],
                    pair["rejected"],
                    pair.get("judge_result_id"),
                    json.dumps(pair.get("scores") or {}),
                    pair.get("overall_score"),
                    pair.get("confidence"),
                    pair.get("category"),
                    pair.get("source", "judge"),
                    json.dumps(meta),
                    pair.get("created_at") or datetime.now(timezone.utc).isoformat(),
                    status,
                    pair.get("quarantine_reason"),
                ),
            )
        return pair["id"]

    def set_preference_training_status(
        self, pref_id: str, status: str, reason: str | None = None
    ) -> None:
        with get_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT meta_json FROM preferences WHERE id = ?", (pref_id,)
            ).fetchone()
            if not row:
                return
            meta = json.loads(row["meta_json"] or "{}")
            meta["training_status"] = status
            if reason is not None:
                meta["quarantine_reason"] = reason
            conn.execute(
                """
                UPDATE preferences
                SET training_status = ?, quarantine_reason = ?, meta_json = ?
                WHERE id = ?
                """,
                (status, reason, json.dumps(meta), pref_id),
            )

    def list_preferences(self, limit: int = 200, query: str | None = None) -> list[dict]:
        with get_conn(self.db_path) as conn:
            if query:
                rows = conn.execute(
                    """
                    SELECT * FROM preferences
                    WHERE prompt LIKE ? OR chosen LIKE ? OR rejected LIKE ? OR category LIKE ?
                    ORDER BY created_at DESC LIMIT ?
                    """,
                    (f"%{query}%", f"%{query}%", f"%{query}%", f"%{query}%", limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM preferences ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["scores"] = json.loads(d.pop("scores_json") or "{}")
            meta = json.loads(d.pop("meta_json") or "{}")
            # Column values win over stale meta copies
            col_status = d.get("training_status")
            col_reason = d.get("quarantine_reason")
            d.update({k: v for k, v in meta.items() if v is not None})
            if col_status:
                d["training_status"] = col_status
            if col_reason is not None:
                d["quarantine_reason"] = col_reason
            out.append(d)
        return out

    def list_training_preferences(self, limit: int = 10000) -> list[dict]:
        """Preferences eligible for DPO after quarantine + quality gates."""
        from src.judge.preference import assess_stored_preference_for_training

        prefs = self.list_preferences(limit=limit)
        usable = []
        for p in prefs:
            if (p.get("training_status") or "").lower() == "quarantined":
                continue
            gate = assess_stored_preference_for_training(p)
            if gate.get("eligible"):
                p = dict(p)
                p["_training_gate"] = gate
                usable.append(p)
        return usable

    def preference_count(self) -> int:
        with get_conn(self.db_path) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM preferences").fetchone()[0])

    def list_evaluations(self, limit: int = 50) -> list[dict]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM evaluations ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["result"] = json.loads(d.pop("result_json"))
            out.append(d)
        return out

    def export_preferences_jsonl(self, path: Path) -> int:
        prefs = self.list_preferences(limit=10000)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for p in prefs:
                rec = {
                    "prompt": p["prompt"],
                    "chosen": p["chosen"],
                    "rejected": p["rejected"],
                    "id": p["id"],
                    "overall_score": p.get("overall_score"),
                    "category": p.get("category"),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return len(prefs)

    def save_experiment(self, exp: dict[str, Any]) -> str:
        now = datetime.now(timezone.utc).isoformat()
        with get_conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments
                (id, name, method, config_json, status, metrics_json,
                 checkpoint_path, log_text, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exp["id"],
                    exp.get("name"),
                    exp.get("method"),
                    json.dumps(exp.get("config") or {}),
                    exp.get("status", "created"),
                    json.dumps(exp.get("metrics") or {}),
                    exp.get("checkpoint_path"),
                    exp.get("log_text", ""),
                    exp.get("created_at", now),
                    now,
                ),
            )
        return exp["id"]

    def latest_experiment(self) -> dict | None:
        with get_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM experiments ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["config"] = json.loads(d.pop("config_json") or "{}")
        d["metrics"] = json.loads(d.pop("metrics_json") or "{}")
        return d

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO conversation_turns (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, datetime.now(timezone.utc).isoformat()),
            )

    def get_history(self, session_id: str, limit: int = 20) -> list[dict]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at FROM conversation_turns
                WHERE session_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]
