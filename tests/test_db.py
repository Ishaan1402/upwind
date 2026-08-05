"""Tests for SQLite narrative/verdict persistence."""

import json
import sqlite3

from backend import db as db_module


def test_update_cached_verdict_persists(monkeypatch, tmp_path):
    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(db_path))
    db_module.init_db()
    db_module.set_cached_narrative("why_k", "narrative", {"signals": []})
    assert db_module.get_cached_narrative("why_k") == "narrative"

    db_module.update_cached_verdict("why_k", {"verdict": "pass", "reasoning": "ok"})
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT judge_verdict_json FROM narrative_cache WHERE cache_key = 'why_k'"
    ).fetchone()
    conn.close()
    assert json.loads(row[0]) == {"verdict": "pass", "reasoning": "ok"}


def test_update_cached_verdict_missing_key_is_noop(monkeypatch, tmp_path):
    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(db_path))
    db_module.init_db()
    db_module.update_cached_verdict("missing", {"verdict": "pass"})  # must not raise
