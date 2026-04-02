"""
Persistent SQLite store for run metrics.
Used by the Streamlit dashboard to show history charts.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .tracker import RunMetrics


class MetricsStore:
    def __init__(self, db_path: str = "./metrics.db") -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id      TEXT PRIMARY KEY,
                    query       TEXT,
                    status      TEXT,
                    confidence  REAL,
                    latency_ms  REAL,
                    total_tokens INTEGER,
                    total_cost  REAL,
                    a2a_count   INTEGER,
                    agents_json TEXT,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def save(self, run: RunMetrics) -> None:
        d = run.to_dict()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                  (run_id, query, status, confidence, latency_ms,
                   total_tokens, total_cost, a2a_count, agents_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    d["run_id"],
                    d["query"],
                    d["status"],
                    d["confidence_score"],
                    d["total_latency_ms"],
                    d["total_tokens"],
                    d["total_cost_usd"],
                    d["a2a_message_count"],
                    json.dumps(d["agents"]),
                ),
            )

    def fetch_all(self) -> list[dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
        return [dict(r) for r in rows]

    def fetch_run(self, run_id: str) -> dict | None:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row:
            d = dict(row)
            d["agents"] = json.loads(d.pop("agents_json", "{}"))
            return d
        return None
