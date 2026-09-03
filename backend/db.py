import sqlite3
import os
import json
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Iterator, Optional, Dict, Any

DB_PATH = os.path.join(os.path.dirname(__file__), "cache.db")


@contextmanager
def _conn(db_path: Optional[str] = None) -> Iterator[sqlite3.Connection]:
    """Yield a connection that is always closed, even on exception."""
    conn = sqlite3.connect(db_path or DB_PATH, timeout=5.0)
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Optional[str] = None):
    with _conn(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS narrative_cache (
                cache_key TEXT PRIMARY KEY,
                narrative TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                judge_verdict_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_narrative_cache_created_at ON narrative_cache(created_at)
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS geocode_cache (
                query_normalized TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS request_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                method TEXT NOT NULL,
                status INTEGER NOT NULL,
                duration_ms REAL NOT NULL
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_request_events_ts ON request_events(ts)
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS why_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                cache_hit INTEGER NOT NULL DEFAULT 0,
                llm_generated INTEGER NOT NULL DEFAULT 0,
                llm_cost_usd REAL NOT NULL DEFAULT 0,
                judge_verdict TEXT,
                country_code TEXT,
                loc_key TEXT,
                total_ms REAL,
                top_hypothesis TEXT,
                top_confidence TEXT,
                llm_input_tokens INTEGER,
                llm_output_tokens INTEGER,
                judge_input_tokens INTEGER,
                judge_output_tokens INTEGER,
                fallback_used INTEGER NOT NULL DEFAULT 0,
                gatekeeper_retries INTEGER NOT NULL DEFAULT 0
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_why_events_ts ON why_events(ts)
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS place_cache (
                key TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                step TEXT NOT NULL,
                status TEXT NOT NULL,
                duration_ms REAL NOT NULL,
                as_of TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_events_ts ON signal_events(ts)
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                event TEXT NOT NULL,
                detail TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_events_ts ON user_events(ts)
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                kind TEXT NOT NULL,
                detail TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_health_events_ts ON health_events(ts)
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_rollup (
                date TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

        # Safe migrations for pre-existing tables (ignore "duplicate column" errors).
        _migrations = [
            ("narrative_cache", "judge_verdict_json", "TEXT"),
            ("why_events", "country_code", "TEXT"),
            ("why_events", "loc_key", "TEXT"),
            ("why_events", "total_ms", "REAL"),
            ("why_events", "top_hypothesis", "TEXT"),
            ("why_events", "top_confidence", "TEXT"),
            ("why_events", "llm_input_tokens", "INTEGER"),
            ("why_events", "llm_output_tokens", "INTEGER"),
            ("why_events", "judge_input_tokens", "INTEGER"),
            ("why_events", "judge_output_tokens", "INTEGER"),
            ("why_events", "fallback_used", "INTEGER DEFAULT 0"),
            ("why_events", "gatekeeper_retries", "INTEGER DEFAULT 0"),
            ("signal_events", "as_of", "TEXT"),
        ]
        for table, column, ddl in _migrations:
            cursor.execute(f"PRAGMA table_info({table})")
            existing_columns = {row[1] for row in cursor.fetchall()}
            if column in existing_columns:
                continue
            try:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
                conn.commit()
            except Exception as e:
                # Multi-worker startup can race on ALTER TABLE; a losing worker
                # finds the column already added on its next run and skips it.
                print(f"[SQLite Migration Skipped]: {table}.{column}: {e}")


def _parse_created_at(created_at_str: str) -> datetime:
    clean_str = created_at_str.replace("Z", "+00:00") if "Z" in created_at_str else created_at_str
    dt = datetime.fromisoformat(clean_str)
    return dt.replace(tzinfo=None)


def get_cached_narrative(cache_key: str, max_age_hours: int = 1) -> Optional[str]:
    try:
        with _conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT narrative, created_at FROM narrative_cache WHERE cache_key = ?", (cache_key,))
            row = cursor.fetchone()

        if not row:
            return None

        narrative, created_at_str = row
        created_at = _parse_created_at(created_at_str)
        if datetime.now(timezone.utc).replace(tzinfo=None) - created_at < timedelta(hours=max_age_hours):
            return narrative
    except Exception as e:
        print(f"[SQLite Cache Read Error]: {e}")
    return None


def set_cached_narrative(cache_key: str, narrative: str, payload: Dict[str, Any], judge_verdict: Optional[Dict[str, Any]] = None):
    try:
        now_str = datetime.now(timezone.utc).isoformat()
        with _conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO narrative_cache (cache_key, narrative, payload_json, judge_verdict_json, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (cache_key, narrative, json.dumps(payload), json.dumps(judge_verdict) if judge_verdict else None, now_str))
            conn.commit()
    except Exception as e:
        print(f"[SQLite Cache Write Error]: {e}")


def update_cached_verdict(cache_key: str, judge_verdict: Optional[Dict[str, Any]]):
    """Persist (or update) a judge verdict for an already-cached narrative."""
    try:
        with _conn() as conn:
            conn.execute(
                "UPDATE narrative_cache SET judge_verdict_json = ? WHERE cache_key = ?",
                (json.dumps(judge_verdict) if judge_verdict else None, cache_key),
            )
            conn.commit()
    except Exception as e:
        print(f"[SQLite Cache Write Error]: {e}")


def get_cached_geocode(query_normalized: str, max_age_days: int = 7) -> Optional[Dict[str, Any]]:
    try:
        with _conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT payload_json, created_at FROM geocode_cache WHERE query_normalized = ?", (query_normalized,))
            row = cursor.fetchone()

        if not row:
            return None

        payload_json, created_at_str = row
        created_at = _parse_created_at(created_at_str)
        if datetime.now(timezone.utc).replace(tzinfo=None) - created_at < timedelta(days=max_age_days):
            return json.loads(payload_json)
    except Exception as e:
        print(f"[SQLite Geocode Cache Read Error]: {e}")
    return None


def set_cached_geocode(query_normalized: str, payload: Dict[str, Any]):
    try:
        now_str = datetime.now(timezone.utc).isoformat()
        with _conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO geocode_cache (query_normalized, payload_json, created_at)
                VALUES (?, ?, ?)
            """, (query_normalized, json.dumps(payload), now_str))
            conn.commit()
    except Exception as e:
        print(f"[SQLite Geocode Cache Write Error]: {e}")


def get_cached_place(key: str, max_age_days: int = 365) -> Optional[Dict[str, Any]]:
    """Read a cached place-context payload (e.g. ZCTA population under pop:{zip})."""
    try:
        with _conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT payload_json, created_at FROM place_cache WHERE key = ?", (key,))
            row = cursor.fetchone()

        if not row:
            return None

        payload_json, created_at_str = row
        created_at = _parse_created_at(created_at_str)
        if datetime.now(timezone.utc).replace(tzinfo=None) - created_at < timedelta(days=max_age_days):
            return json.loads(payload_json)
    except Exception as e:
        print(f"[SQLite Place Cache Read Error]: {e}")
    return None


def set_cached_place(key: str, payload: Dict[str, Any]):
    try:
        now_str = datetime.now(timezone.utc).isoformat()
        with _conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO place_cache (key, payload_json, created_at)
                VALUES (?, ?, ?)
            """, (key, json.dumps(payload), now_str))
            conn.commit()
    except Exception as e:
        print(f"[SQLite Place Cache Write Error]: {e}")
