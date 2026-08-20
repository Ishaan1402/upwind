"""Persisted request and explanation metrics for Upwind.

The HTTP middleware records every request into ``request_events`` and the
Why routers record cache/cost/judge details into ``why_events``. The
``report`` command turns those rows into the JSON that is embedded in the
eval dashboard:

    python -m backend.metrics report --days 30 --out metrics.json
"""

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from backend.config import LLM_INPUT_PRICE_PER_1M, LLM_OUTPUT_PRICE_PER_1M
from backend.db import DB_PATH, init_db


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def record_request(
    method: str,
    path: str,
    status: int,
    duration_ms: float,
    db_path: str = DB_PATH,
) -> None:
    """Record one HTTP request. Never raises; observability must not break the app."""
    try:
        conn = _connect(db_path)
        conn.execute(
            "INSERT INTO request_events (ts, endpoint, method, status, duration_ms) "
            "VALUES (?, ?, ?, ?, ?)",
            (_utcnow_iso(), path, method, int(status), round(float(duration_ms), 2)),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def record_why(
    endpoint: str,
    cache_hit: bool = False,
    llm_generated: bool = False,
    llm_cost_usd: float = 0.0,
    judge_verdict: Optional[str] = None,
    country_code: Optional[str] = None,
    db_path: str = DB_PATH,
) -> None:
    """Record one Why explanation (cache hit, cost, judge result)."""
    try:
        conn = _connect(db_path)
        conn.execute(
            "INSERT INTO why_events "
            "(ts, endpoint, cache_hit, llm_generated, llm_cost_usd, judge_verdict, country_code) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                _utcnow_iso(),
                endpoint,
                int(bool(cache_hit)),
                int(bool(llm_generated)),
                round(float(llm_cost_usd), 8),
                judge_verdict,
                country_code,
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def estimate_llm_cost(
    text: str,
    input_tokens: int = 0,
    output_tokens: Optional[int] = None,
) -> float:
    """Estimate LLM cost from provider usage when available, otherwise tokens≈chars/4."""
    completion_tokens = output_tokens if output_tokens is not None else max(1, len(text) // 4)
    prompt_cost = (input_tokens or 0) / 1_000_000 * LLM_INPUT_PRICE_PER_1M
    completion_cost = completion_tokens / 1_000_000 * LLM_OUTPUT_PRICE_PER_1M
    return round(prompt_cost + completion_cost, 8)


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    index = max(0, math.ceil(p * len(ordered)) - 1)
    return round(ordered[index], 2)


def _load_rows(db_path: str, table: str) -> List[sqlite3.Row]:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(f"SELECT * FROM {table}").fetchall()
    conn.close()
    return rows


def report(days: int = 30, db_path: str = DB_PATH) -> Dict[str, Any]:
    """Aggregate the last ``days`` of events into a compact metrics document."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    request_rows = conn.execute(
        "SELECT endpoint, method, status, duration_ms, ts FROM request_events "
        "WHERE datetime(ts) >= datetime(?)",
        (cutoff,),
    ).fetchall()
    why_rows = conn.execute(
        "SELECT endpoint, cache_hit, llm_generated, llm_cost_usd, judge_verdict, country_code, ts "
        "FROM why_events WHERE datetime(ts) >= datetime(?)",
        (cutoff,),
    ).fetchall()
    verdict_rows = conn.execute(
        "SELECT judge_verdict_json FROM narrative_cache "
        "WHERE judge_verdict_json IS NOT NULL AND datetime(created_at) >= datetime(?)",
        (cutoff,),
    ).fetchall()
    narrative_rows = conn.execute(
        "SELECT narrative FROM narrative_cache "
        "WHERE datetime(created_at) >= datetime(?)",
        (cutoff,),
    ).fetchall()
    conn.close()

    # Narrative verbosity: LLM briefings must stay tight (~110 words). Track
    # the distribution so the dashboard can catch verbosity drift.
    word_counts = sorted(
        len((row["narrative"] or "").split()) for row in narrative_rows
    )
    narrative_count = len(word_counts)
    narratives: Dict[str, Any] = {
        "count": narrative_count,
        "avg_words": round(sum(word_counts) / narrative_count, 1) if narrative_count else None,
        "median_words": _percentile(word_counts, 0.50) if narrative_count else None,
        "p90_words": _percentile(word_counts, 0.90) if narrative_count else None,
        "pct_over_150_words": (
            round(100 * sum(1 for w in word_counts if w > 150) / narrative_count, 1)
            if narrative_count else None
        ),
    }

    request_count = len(request_rows)
    endpoint_status_counts: Dict[str, int] = Counter()
    daily_counts: Dict[str, int] = Counter()
    latency: Dict[str, List[float]] = defaultdict(list)
    for row in request_rows:
        key = f"{row['endpoint']} {row['method']}"
        endpoint_status_counts[f"{key} {row['status']}"] += 1
        daily_counts[str(row["ts"])[:10]] += 1
        latency[key].append(row["duration_ms"])

    latency_summary = {
        key: {
            "count": len(values),
            "p50_ms": _percentile(values, 0.50),
            "p95_ms": _percentile(values, 0.95),
            "max_ms": round(max(values), 2),
        }
        for key, values in sorted(latency.items())
    }

    why_total = len(why_rows)
    cache_hits = sum(1 for r in why_rows if r["cache_hit"])
    llm_generated = sum(1 for r in why_rows if r["llm_generated"])
    llm_cost_usd = sum(float(r["llm_cost_usd"] or 0) for r in why_rows)
    judge_counts = Counter(
        r["judge_verdict"] for r in why_rows if r["judge_verdict"]
    )
    cached_judge_counts: Dict[str, int] = Counter()
    for row in verdict_rows:
        try:
            verdict = json.loads(row["judge_verdict_json"])
            value = verdict.get("verdict")
            if value in ("pass", "fail", "unknown", "skipped"):
                cached_judge_counts[value] += 1
        except Exception:
            continue

    judged = judge_counts.get("pass", 0) + judge_counts.get("fail", 0)
    cached_judged = cached_judge_counts.get("pass", 0) + cached_judge_counts.get("fail", 0)

    return {
        "generated_at": _utcnow_iso(),
        "window_days": days,
        "requests": {
            "total": request_count,
            "by_endpoint_status": dict(sorted(endpoint_status_counts.items())),
            "daily": dict(sorted(daily_counts.items())),
            "latency": latency_summary,
        },
        "why": {
            "total": why_total,
            "cache_hit_rate": round(cache_hits / why_total, 4) if why_total else None,
            "cache_hits": cache_hits,
            "llm_generated": llm_generated,
            "llm_cost_per_explanation_usd": round(llm_cost_usd / why_total, 6) if why_total else None,
            "llm_cost_usd": round(llm_cost_usd, 6),
            "judge": dict(sorted(judge_counts.items())),
            "judge_pass_rate": round(judge_counts["pass"] / judged, 4) if judged else None,
            "cached_judge": dict(sorted(cached_judge_counts.items())),
            "cached_judge_pass_rate": (
                round(cached_judge_counts["pass"] / cached_judged, 4) if cached_judged else None
            ),
        },
        "narratives": narratives,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="backend.metrics", description=__doc__)
    parser.add_argument("--db", default=DB_PATH, help="SQLite path (default: backend/cache.db)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    report_parser = subparsers.add_parser("report", help="aggregate metrics into JSON")
    report_parser.add_argument("--days", type=int, default=30, help="lookback window in days")
    report_parser.add_argument("--out", help="optional JSON output path")

    args = parser.parse_args(argv)
    init_db(args.db)
    result = report(days=args.days, db_path=args.db)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote metrics to {args.out}")
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
