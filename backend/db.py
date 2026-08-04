import sqlite3
import os
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any

DB_PATH = os.path.join(os.path.dirname(__file__), "cache.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
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
        CREATE TABLE IF NOT EXISTS geocode_cache (
            query_normalized TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    
    # Safe migration if table existed without judge_verdict_json
    try:
        cursor.execute("ALTER TABLE narrative_cache ADD COLUMN judge_verdict_json TEXT")
        conn.commit()
    except Exception:
        pass
        
    conn.close()

def _parse_created_at(created_at_str: str) -> datetime:
    clean_str = created_at_str.replace("Z", "+00:00") if "Z" in created_at_str else created_at_str
    dt = datetime.fromisoformat(clean_str)
    return dt.replace(tzinfo=None)

def get_cached_narrative(cache_key: str, max_age_hours: int = 1) -> Optional[str]:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT narrative, created_at FROM narrative_cache WHERE cache_key = ?", (cache_key,))
        row = cursor.fetchone()
        conn.close()

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
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        now_str = datetime.now(timezone.utc).isoformat()
        cursor.execute("""
            INSERT OR REPLACE INTO narrative_cache (cache_key, narrative, payload_json, judge_verdict_json, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (cache_key, narrative, json.dumps(payload), json.dumps(judge_verdict) if judge_verdict else None, now_str))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[SQLite Cache Write Error]: {e}")

def update_cached_verdict(cache_key: str, judge_verdict: Optional[Dict[str, Any]]):
    """Persist (or update) a judge verdict for an already-cached narrative."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE narrative_cache SET judge_verdict_json = ? WHERE cache_key = ?",
            (json.dumps(judge_verdict) if judge_verdict else None, cache_key),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[SQLite Cache Write Error]: {e}")

def get_cached_geocode(query_normalized: str, max_age_days: int = 7) -> Optional[Dict[str, Any]]:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT payload_json, created_at FROM geocode_cache WHERE query_normalized = ?", (query_normalized,))
        row = cursor.fetchone()
        conn.close()

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
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        now_str = datetime.now(timezone.utc).isoformat()
        cursor.execute("""
            INSERT OR REPLACE INTO geocode_cache (query_normalized, payload_json, created_at)
            VALUES (?, ?, ?)
        """, (query_normalized, json.dumps(payload), now_str))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[SQLite Geocode Cache Write Error]: {e}")
