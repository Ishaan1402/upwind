"""Persisted request and explanation metrics for Upwind.

The HTTP middleware records every request into ``request_events``, the Why
routers record cache/cost/judge details into ``why_events`` plus one row per
evidence tool step into ``signal_events``, and the frontend pings
``/api/events`` for user-behavior events. The ``report`` command turns the
last N days of rows into the five-axis metrics JSON rendered on the public
evidence page:

    python -m backend.metrics report --days 30 --out metrics.json

Raw event rows are pruned after ``METRICS_RETENTION_DAYS`` (default 90).
Before pruning, every completed UTC day is rolled into ``daily_rollup`` so
older history survives the prune in aggregated form (counts + sums only):

    python -m backend.metrics rollup
    python -m backend.metrics prune
"""

import argparse
import json
import math
import sqlite3
from contextlib import contextmanager
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Iterator, List, Optional

from backend.config import (
    LLM_INPUT_PRICE_PER_1M,
    LLM_OUTPUT_PRICE_PER_1M,
    LLM_JUDGE_INPUT_PRICE_PER_1M,
    LLM_JUDGE_OUTPUT_PRICE_PER_1M,
    METRICS_RETENTION_DAYS,
)
from backend.db import DB_PATH, init_db


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _db(db_path: str, timeout: float = 5.0) -> Iterator[sqlite3.Connection]:
    """Yield a connection that is always closed, even on exception."""
    conn = sqlite3.connect(db_path, timeout=timeout)
    try:
        yield conn
    finally:
        conn.close()


def _record_failure(db_path: str, kind: str, detail: str) -> None:
    """Best-effort write to health_events; never raises.

    Uses a short timeout so a lock-contended primary write never cascades into
    a second 5-second block here.
    """
    try:
        with _db(db_path, timeout=0.5) as conn:
            conn.execute(
                "INSERT INTO health_events (ts, kind, detail) VALUES (?, ?, ?)",
                (_utcnow_iso(), kind, (detail or "")[:500]),
            )
            conn.commit()
    except Exception:
        pass


def record_request(
    method: str,
    path: str,
    status: int,
    duration_ms: float,
    db_path: str = DB_PATH,
) -> None:
    """Record one HTTP request. Never raises; observability must not break the app."""
    try:
        with _db(db_path) as conn:
            conn.execute(
                "INSERT INTO request_events (ts, endpoint, method, status, duration_ms) "
                "VALUES (?, ?, ?, ?, ?)",
                (_utcnow_iso(), path, method, int(status), round(float(duration_ms), 2)),
            )
            conn.commit()
    except Exception as e:
        _record_failure(db_path, "db_write", f"request_events: {e}")


def record_why(
    endpoint: str,
    cache_hit: bool = False,
    llm_generated: bool = False,
    llm_cost_usd: float = 0.0,
    judge_verdict: Optional[str] = None,
    country_code: Optional[str] = None,
    loc_key: Optional[str] = None,
    total_ms: Optional[float] = None,
    top_hypothesis: Optional[str] = None,
    top_confidence: Optional[str] = None,
    llm_input_tokens: Optional[int] = None,
    llm_output_tokens: Optional[int] = None,
    judge_input_tokens: Optional[int] = None,
    judge_output_tokens: Optional[int] = None,
    fallback_used: bool = False,
    gatekeeper_retries: int = 0,
    db_path: str = DB_PATH,
) -> Optional[int]:
    """Record one Why explanation; returns the new row id (None on failure).

    The row id lets the async stream judge write its verdict back to the
    exact row after the response is already streamed.
    """
    try:
        with _db(db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO why_events "
                "(ts, endpoint, cache_hit, llm_generated, llm_cost_usd, judge_verdict, "
                " country_code, loc_key, total_ms, top_hypothesis, top_confidence, "
                " llm_input_tokens, llm_output_tokens, judge_input_tokens, judge_output_tokens, "
                " fallback_used, gatekeeper_retries) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _utcnow_iso(),
                    endpoint,
                    int(bool(cache_hit)),
                    int(bool(llm_generated)),
                    round(float(llm_cost_usd), 8),
                    judge_verdict,
                    country_code,
                    loc_key,
                    round(float(total_ms), 2) if total_ms is not None else None,
                    top_hypothesis,
                    top_confidence,
                    int(llm_input_tokens) if llm_input_tokens is not None else None,
                    int(llm_output_tokens) if llm_output_tokens is not None else None,
                    int(judge_input_tokens) if judge_input_tokens is not None else None,
                    int(judge_output_tokens) if judge_output_tokens is not None else None,
                    int(bool(fallback_used)),
                    int(gatekeeper_retries or 0),
                ),
            )
            row_id = cursor.lastrowid
            conn.commit()
        return int(row_id)
    except Exception as e:
        _record_failure(db_path, "db_write", f"why_events: {e}")
        return None


def update_why_verdict(
    why_event_id: Optional[int],
    verdict: Optional[str],
    judge_input_tokens: Optional[int] = None,
    judge_output_tokens: Optional[int] = None,
    db_path: str = DB_PATH,
) -> None:
    """Write an async judge result back to a why_events row. Never raises."""
    if why_event_id is None:
        return
    try:
        with _db(db_path) as conn:
            conn.execute(
                "UPDATE why_events SET judge_verdict = ?, judge_input_tokens = ?, judge_output_tokens = ? "
                "WHERE id = ?",
                (
                    verdict,
                    int(judge_input_tokens) if judge_input_tokens is not None else None,
                    int(judge_output_tokens) if judge_output_tokens is not None else None,
                    int(why_event_id),
                ),
            )
            conn.commit()
    except Exception as e:
        _record_failure(db_path, "db_write", f"update_why_verdict: {e}")


def record_signal_events(endpoint: str, steps: List[Dict[str, Any]], db_path: str = DB_PATH) -> None:
    """Record one row per evidence tool step (status + duration). Never raises."""
    if not steps:
        return
    try:
        now = _utcnow_iso()
        with _db(db_path) as conn:
            conn.executemany(
                "INSERT INTO signal_events (ts, endpoint, step, status, duration_ms, as_of) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        now, endpoint, s.get("step") or "?", s.get("status") or "?",
                        round(float(s.get("duration_ms") or 0), 2), s.get("as_of"),
                    )
                    for s in steps
                ],
            )
            conn.commit()
    except Exception as e:
        _record_failure(db_path, "db_write", f"signal_events: {e}")


def record_user_event(event: str, detail: Optional[str] = None, db_path: str = DB_PATH) -> None:
    """Record a frontend user-behavior event (aqi_view, why_open, ...). Never raises."""
    try:
        with _db(db_path) as conn:
            conn.execute(
                "INSERT INTO user_events (ts, event, detail) VALUES (?, ?, ?)",
                (_utcnow_iso(), event, (detail or "")[:200] or None),
            )
            conn.commit()
    except Exception as e:
        _record_failure(db_path, "db_write", f"user_events: {e}")


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


def _token_cost(
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    input_price: float,
    output_price: float,
) -> Optional[float]:
    """USD cost from token counts when both are present, else None."""
    if input_tokens is None or output_tokens is None:
        return None
    return round(input_tokens / 1_000_000 * input_price + output_tokens / 1_000_000 * output_price, 8)


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    index = max(0, math.ceil(p * len(ordered)) - 1)
    return round(ordered[index], 2)


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 4) if denominator else None


# --------------------------------------------------------------------------- #
# Shared row aggregation. Used by the raw report, the daily rollup and the    #
# hybrid (rollup + raw) report so the three never diverge.                    #
# --------------------------------------------------------------------------- #

def _fetch_raw_rows(conn: sqlite3.Connection, cutoff: str) -> Dict[str, Any]:
    """Fetch raw event rows with ``ts >= cutoff`` plus narrative_cache extras."""
    conn.row_factory = sqlite3.Row
    return {
        "request_rows": conn.execute(
            "SELECT endpoint, method, status, duration_ms, ts FROM request_events "
            "WHERE ts >= ?",
            (cutoff,),
        ).fetchall(),
        "why_rows": conn.execute(
            "SELECT * FROM why_events WHERE ts >= ?",
            (cutoff,),
        ).fetchall(),
        "signal_rows": conn.execute(
            "SELECT step, status, duration_ms, as_of FROM signal_events WHERE ts >= ?",
            (cutoff,),
        ).fetchall(),
        "user_rows": conn.execute(
            "SELECT event FROM user_events WHERE ts >= ?",
            (cutoff,),
        ).fetchall(),
        "health_rows": conn.execute(
            "SELECT kind FROM health_events WHERE ts >= ?",
            (cutoff,),
        ).fetchall(),
        "verdict_rows": conn.execute(
            "SELECT judge_verdict_json FROM narrative_cache "
            "WHERE judge_verdict_json IS NOT NULL AND datetime(created_at) >= datetime(?)",
            (cutoff,),
        ).fetchall(),
        "narrative_rows": conn.execute(
            "SELECT narrative FROM narrative_cache "
            "WHERE datetime(created_at) >= datetime(?)",
            (cutoff,),
        ).fetchall(),
        "cached_total": conn.execute("SELECT COUNT(*) AS n FROM narrative_cache").fetchone()["n"],
    }


def _agg_requests(request_rows) -> Dict[str, Any]:
    """Per-request aggregation shared by the raw report and the daily rollup."""
    endpoint_status_counts: Counter = Counter()
    daily_counts: Counter = Counter()
    hourly_counts: Counter = Counter()
    latency: Dict[str, List[float]] = defaultdict(list)
    all_latency: List[float] = []
    status_counter: Counter = Counter()
    for row in request_rows:
        key = f"{row['endpoint']} {row['method']}"
        endpoint_status_counts[f"{key} {row['status']}"] += 1
        day, hour = str(row["ts"])[:10], str(row["ts"])[:13]
        daily_counts[day] += 1
        hourly_counts[hour] += 1
        latency[key].append(row["duration_ms"])
        all_latency.append(row["duration_ms"])
        status_counter[row["status"]] += 1
    return {
        "request_count": len(request_rows),
        "endpoint_status_counts": endpoint_status_counts,
        "daily_counts": daily_counts,
        "hourly_counts": hourly_counts,
        "latency": latency,
        "all_latency": all_latency,
        "status_counter": status_counter,
    }


def _agg_why(why_rows) -> Dict[str, Any]:
    """Per-why-row aggregation shared by the raw report and the daily rollup."""
    why_total = len(why_rows)
    cache_hits = sum(1 for r in why_rows if r["cache_hit"])
    llm_generated = sum(1 for r in why_rows if r["llm_generated"])
    fallback_count = sum(1 for r in why_rows if r["fallback_used"])
    retry_count = sum(int(r["gatekeeper_retries"] or 0) for r in why_rows)
    why_daily: Counter = Counter()
    wall_ms: List[float] = []
    wall_cache_hit: List[float] = []
    wall_cache_miss: List[float] = []
    loc_keys: set = set()
    coverage_modes: Counter = Counter()
    top_hypotheses: Counter = Counter()
    judge_counts: Counter = Counter()
    llm_in_tokens, llm_out_tokens, judge_in_tokens, judge_out_tokens = 0, 0, 0, 0
    llm_token_rows = 0
    judge_token_rows = 0
    llm_cost_usd_stored = 0.0
    llm_cost_usd_real = 0.0
    judge_cost_usd = 0.0
    stored_estimate_rows: List[float] = []
    for row in why_rows:
        why_daily[str(row["ts"])[:10]] += 1
        if row["total_ms"] is not None:
            wall_ms.append(row["total_ms"])
            (wall_cache_hit if row["cache_hit"] else wall_cache_miss).append(row["total_ms"])
        if row["loc_key"]:
            loc_keys.add(row["loc_key"])
        if row["country_code"]:
            coverage_modes[(row["country_code"] or "").lower()] += 1
        if row["top_hypothesis"]:
            top_hypotheses[row["top_hypothesis"]] += 1
        if row["judge_verdict"]:
            judge_counts[row["judge_verdict"]] += 1
        llm_in_tokens += int(row["llm_input_tokens"] or 0)
        llm_out_tokens += int(row["llm_output_tokens"] or 0)
        judge_in_tokens += int(row["judge_input_tokens"] or 0)
        judge_out_tokens += int(row["judge_output_tokens"] or 0)
        if row["llm_input_tokens"] is not None and row["llm_output_tokens"] is not None:
            llm_token_rows += 1
        if row["judge_input_tokens"] is not None and row["judge_output_tokens"] is not None:
            judge_token_rows += 1
        stored = float(row["llm_cost_usd"] or 0)
        llm_cost_usd_stored += stored
        real = _token_cost(
            row["llm_input_tokens"], row["llm_output_tokens"],
            LLM_INPUT_PRICE_PER_1M, LLM_OUTPUT_PRICE_PER_1M,
        )
        if real is not None:
            llm_cost_usd_real += real
        judge_cost_usd += _token_cost(
            row["judge_input_tokens"], row["judge_output_tokens"],
            LLM_JUDGE_INPUT_PRICE_PER_1M, LLM_JUDGE_OUTPUT_PRICE_PER_1M,
        ) or 0.0
        if row["llm_input_tokens"] is not None and row["llm_output_tokens"] is not None:
            stored_estimate_rows.append(stored)
    return {
        "why_total": why_total,
        "cache_hits": cache_hits,
        "llm_generated": llm_generated,
        "fallback_count": fallback_count,
        "retry_count": retry_count,
        "why_daily": why_daily,
        "wall_ms": wall_ms,
        "wall_cache_hit": wall_cache_hit,
        "wall_cache_miss": wall_cache_miss,
        "loc_keys": loc_keys,
        "coverage_modes": coverage_modes,
        "top_hypotheses": top_hypotheses,
        "judge_counts": judge_counts,
        "llm_in_tokens": llm_in_tokens,
        "llm_out_tokens": llm_out_tokens,
        "judge_in_tokens": judge_in_tokens,
        "judge_out_tokens": judge_out_tokens,
        "llm_token_rows": llm_token_rows,
        "judge_token_rows": judge_token_rows,
        "llm_cost_usd_stored": llm_cost_usd_stored,
        "llm_cost_usd_real": llm_cost_usd_real,
        "judge_cost_usd": judge_cost_usd,
        "stored_estimate_rows": stored_estimate_rows,
    }


def _llm_cost_total(why: Dict[str, Any]) -> float:
    """Prefer real token-derived cost for new rows; old rows (no tokens) keep
    their stored estimate. Sum of estimates on rows that now have tokens is
    subtracted so nothing is double-counted."""
    return why["llm_cost_usd_real"] + max(0.0, why["llm_cost_usd_stored"] - sum(why["stored_estimate_rows"]))


def _agg_signals(signal_rows) -> Dict[str, Dict[str, Any]]:
    """Per-signal-row aggregation (evidence tool steps) shared by report + rollup."""
    now_utc = datetime.now(timezone.utc)
    feed_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "present": 0, "absent": 0, "unavailable": 0, "durations": [], "as_of_ages": []})
    for row in signal_rows:
        s = feed_stats[row["step"]]
        s["count"] += 1
        # Execution-trace statuses: done = feed responded (present),
        # absent = cleanly absent, warning = feed unavailable/failed.
        status = (row["status"] or "").lower()
        if status == "done":
            s["present"] += 1
        elif status == "absent":
            s["absent"] += 1
        else:  # "warning" or anything unexpected
            s["unavailable"] += 1
        s["durations"].append(row["duration_ms"])
        if row["as_of"]:
            try:
                as_of = datetime.fromisoformat(str(row["as_of"]))
                if as_of.tzinfo is None:
                    as_of = as_of.replace(tzinfo=timezone.utc)
                s["as_of_ages"].append(max(0.0, (now_utc - as_of).total_seconds() / 3600))
            except (TypeError, ValueError):
                continue
    for s in feed_stats.values():
        s["max_age_hours"] = round(max(s["as_of_ages"]), 1) if s["as_of_ages"] else None
        s["as_of_count"] = len(s["as_of_ages"])
        del s["as_of_ages"]
    return feed_stats


def _agg_verdicts(verdict_rows) -> Counter:
    """Parse cached narrative verdicts shared by report + rollup."""
    cached_judge_counts: Counter = Counter()
    for row in verdict_rows:
        try:
            verdict = json.loads(row["judge_verdict_json"])
            value = verdict.get("verdict")
            if value in ("pass", "fail", "unknown", "skipped"):
                cached_judge_counts[value] += 1
        except Exception:
            continue
    return cached_judge_counts


def _agg_narratives(narrative_rows) -> List[int]:
    """Sorted per-narrative word counts shared by report + rollup."""
    return sorted(len((row["narrative"] or "").split()) for row in narrative_rows)


def _latency_summary(latency: Dict[str, List[float]]) -> Dict[str, Any]:
    """Per-endpoint latency summary in the shape used by report + rollup."""
    return {
        key: {
            "count": len(values),
            "p50_ms": _percentile(values, 0.50),
            "p95_ms": _percentile(values, 0.95),
            "p99_ms": _percentile(values, 0.99),
            "max_ms": round(max(values), 2),
        }
        for key, values in sorted(latency.items())
    }


def _weighted_mean(units: List[Dict[str, Any]], value_key: str, weight_key: str = "count", nd: int = 2) -> Optional[float]:
    """Count-weighted mean of ``value_key`` across summary units; None if no weights.

    Used by the hybrid report to approximate merged percentiles from per-day
    rollup summaries, where exact global percentile merging is impossible.
    """
    pairs = [(u[value_key], u[weight_key]) for u in units if u.get(value_key) is not None and u.get(weight_key)]
    total_weight = sum(w for _, w in pairs)
    if not total_weight:
        return None
    return round(sum(v * w for v, w in pairs) / total_weight, nd)


def _merge_latency_units(units: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Best-effort merge of per-day latency summaries (count + p50/p95/p99/max)."""
    merged: Dict[str, Any] = {"count": sum(u.get("count", 0) for u in units)}
    for p in ("p50_ms", "p95_ms", "p99_ms"):
        merged[p] = _weighted_mean(units, p)
    maxes = [u["max_ms"] for u in units if u.get("count") and u.get("max_ms") is not None]
    merged["max_ms"] = round(max(maxes), 2) if maxes else None
    return merged


def report(days: int = 30, db_path: str = DB_PATH) -> Dict[str, Any]:
    """Aggregate the last ``days`` of events into a five-axis metrics document.

    Windows at or under ``METRICS_RETENTION_DAYS`` are computed entirely from
    raw event rows (unchanged behavior, identical output shape). Larger windows
    merge the daily rollup for the pruned range ``[now-days, now-retention)``
    with raw rows for the active range ``[now-retention, now)`` and mark the
    result with ``rolled_up: True`` and ``rolled_up_days``. Additive metrics
    (counts/sums) merge exactly; latency percentiles over the merged window are
    approximate because per-day percentiles cannot reproduce an exact global
    percentile, so they are best-effort count-weighted estimates flagged with
    ``performance.percentiles_note``.
    """
    if days <= METRICS_RETENTION_DAYS:
        return _report_raw(days, db_path)
    return _report_hybrid(days, db_path, METRICS_RETENTION_DAYS)


def _report_raw(days: int, db_path: str) -> Dict[str, Any]:
    # Event tables store ISO 8601 UTC (lexicographically sortable), so compare
    # the ts column directly to keep the B-tree index seekable. narrative_cache
    # uses SQLite CURRENT_TIMESTAMP ("YYYY-MM-DD HH:MM:SS"), so its created_at
    # is compared via datetime() for format normalization (it is unindexed).
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    with _db(db_path) as conn:
        rows = _fetch_raw_rows(conn, cutoff)

    req = _agg_requests(rows["request_rows"])
    why = _agg_why(rows["why_rows"])
    feed_stats = _agg_signals(rows["signal_rows"])
    cached_judge_counts = _agg_verdicts(rows["verdict_rows"])
    word_counts = _agg_narratives(rows["narrative_rows"])
    user_event_counts = Counter(r["event"] for r in rows["user_rows"])
    db_write_failures = sum(1 for r in rows["health_rows"] if r["kind"] == "db_write")

    # ---------------------------------------------------------------- requests
    request_count = req["request_count"]
    latency_summary = _latency_summary(req["latency"])
    peak_hour = max(req["hourly_counts"].items(), key=lambda kv: kv[1]) if req["hourly_counts"] else ("", 0)
    five_xx = sum(n for s, n in req["status_counter"].items() if s >= 500)
    err_503 = req["status_counter"].get(503, 0)
    err_429 = req["status_counter"].get(429, 0)

    # ------------------------------------------------------------- why events
    why_total = why["why_total"]
    cache_hits = why["cache_hits"]
    llm_generated = why["llm_generated"]
    fallback_count = why["fallback_count"]
    retry_count = why["retry_count"]
    llm_cost_total = _llm_cost_total(why)
    judge_cost_usd = why["judge_cost_usd"]
    judged = why["judge_counts"].get("pass", 0) + why["judge_counts"].get("fail", 0)
    cached_judged = cached_judge_counts.get("pass", 0) + cached_judge_counts.get("fail", 0)

    # -------------------------------------------------------- signal (feeds)
    feeds: Dict[str, Any] = {}
    for step, s in sorted(feed_stats.items()):
        total = max(1, s["count"])
        feeds[step] = {
            "count": s["count"],
            "present": s["present"],
            "absent": s["absent"],
            "unavailable": s["unavailable"],
            "availability_pct": round(100 * (s["present"] + s["absent"]) / total, 1),
            "p50_ms": _percentile(s["durations"], 0.50),
            "p95_ms": _percentile(s["durations"], 0.95),
            "max_age_hours": s["max_age_hours"],
            "as_of_count": s["as_of_count"],
        }

    # -------------------------------------------------------------- quality
    narrative_count = len(word_counts)

    # ----------------------------------------------------------------- report
    return {
        "generated_at": _utcnow_iso(),
        "window_days": days,
        "requests": {
            "total": request_count,
            "by_endpoint_status": dict(sorted(req["endpoint_status_counts"].items())),
            "daily": dict(sorted(req["daily_counts"].items())),
            "latency": latency_summary,
        },
        "why": {
            "total": why_total,
            "cache_hit_rate": _rate(cache_hits, why_total),
            "cache_hits": cache_hits,
            "llm_generated": llm_generated,
            "llm_cost_per_explanation_usd": round(llm_cost_total / why_total, 6) if why_total else None,
            "llm_cost_usd": round(llm_cost_total, 6),
            "judge": dict(sorted(why["judge_counts"].items())),
            "judge_pass_rate": _rate(why["judge_counts"]["pass"], judged),
            "cached_judge": dict(sorted(cached_judge_counts.items())),
            "cached_judge_pass_rate": _rate(cached_judge_counts["pass"], cached_judged),
        },
        "narratives": {
            "count": narrative_count,
            "avg_words": round(sum(word_counts) / narrative_count, 1) if narrative_count else None,
            "median_words": _percentile(word_counts, 0.50) if narrative_count else None,
            "p90_words": _percentile(word_counts, 0.90) if narrative_count else None,
            "pct_over_150_words": (
                round(100 * sum(1 for w in word_counts if w > 150) / narrative_count, 1)
                if narrative_count else None
            ),
        },
        "scale": {
            "requests_total": request_count,
            "requests_daily": dict(sorted(req["daily_counts"].items())),
            "why_total": why_total,
            "why_daily": dict(sorted(why["why_daily"].items())),
            "distinct_locations": len(why["loc_keys"]),
            "coverage_modes": dict(sorted(why["coverage_modes"].items())),
            "narratives_cached_total": rows["cached_total"],
            "user_events": dict(sorted(user_event_counts.items())),
        },
        "performance": {
            "endpoints": latency_summary,
            "overall": {
                "count": request_count,
                "p50_ms": _percentile(req["all_latency"], 0.50),
                "p95_ms": _percentile(req["all_latency"], 0.95),
                "p99_ms": _percentile(req["all_latency"], 0.99),
                "max_ms": round(max(req["all_latency"]), 2) if req["all_latency"] else None,
            },
            "peak_req_per_hour": {"hour": peak_hour[0], "count": peak_hour[1]},
            "why_wall_ms": {
                "avg_ms": round(sum(why["wall_ms"]) / len(why["wall_ms"]), 1) if why["wall_ms"] else None,
                "p95_ms": _percentile(why["wall_ms"], 0.95),
            },
            "cache_split_ms": {
                "cache_hit_p95_ms": _percentile(why["wall_cache_hit"], 0.95),
                "cache_miss_p95_ms": _percentile(why["wall_cache_miss"], 0.95),
            },
            "steps": feeds,
        },
        "quality": {
            "judge": dict(sorted(why["judge_counts"].items())),
            "judge_pass_rate": _rate(why["judge_counts"]["pass"], judged),
            "fallback_count": fallback_count,
            "fallback_rate": _rate(fallback_count, why_total),
            "gatekeeper_retry_count": retry_count,
            "gatekeeper_retry_rate": _rate(retry_count, llm_generated),
            "top_hypotheses": dict(sorted(why["top_hypotheses"].items(), key=lambda kv: -kv[1])),
            "narrative_length": {
                "avg_words": round(sum(word_counts) / narrative_count, 1) if narrative_count else None,
                "pct_over_150_words": (
                    round(100 * sum(1 for w in word_counts if w > 150) / narrative_count, 1)
                    if narrative_count else None
                ),
            },
            "agreement": None,  # populated once agent/human labels exist on live narratives
        },
        "efficiency": {
            "cache_hit_rate": _rate(cache_hits, why_total),
            "cache_hits": cache_hits,
            "cache_misses": why_total - cache_hits,
            "llm_cost_usd": round(llm_cost_total, 6),
            "llm_cost_per_explanation_usd": round(llm_cost_total / why_total, 6) if why_total else None,
            "judge_cost_usd": round(judge_cost_usd, 6),
            "judge_cost_per_explanation_usd": round(judge_cost_usd / why_total, 6) if why_total else None,
            "avg_input_tokens": round(why["llm_in_tokens"] / why["llm_token_rows"], 1) if why["llm_token_rows"] else None,
            "avg_output_tokens": round(why["llm_out_tokens"] / why["llm_token_rows"], 1) if why["llm_token_rows"] else None,
            "avg_judge_tokens": (
                round((why["judge_in_tokens"] + why["judge_out_tokens"]) / why["judge_token_rows"], 1)
                if why["judge_token_rows"] else None
            ),
            "cost_saved_by_cache_usd": (
                round(cache_hits * (llm_cost_total / max(1, why_total - cache_hits)), 6)
                if why_total - cache_hits else None
            ),
        },
        "reliability": {
            "feeds": feeds,
            "http": {
                "5xx_count": five_xx,
                "5xx_rate": _rate(five_xx, request_count),
                "503_count": err_503,
                "503_rate": _rate(err_503, request_count),
                "429_count": err_429,
                "429_rate": _rate(err_429, request_count),
                "uptime_proxy_pct": (
                    round(100 * (request_count - five_xx) / request_count, 2) if request_count else None
                ),
            },
            "health": {"db_write_failures": db_write_failures},
        },
    }


def rollup_day(date_str: str, db_path: str = DB_PATH) -> None:
    """Aggregate ONE UTC calendar day (``YYYY-MM-DD``) into ``daily_rollup``.

    Query the five raw event tables by ``substr(ts,1,10) = date`` and store
    counts/sums (plus per-day latency percentiles) as a JSON payload via
    ``INSERT OR REPLACE``. Re-runnable and deterministic for a given day.
    """
    with _db(db_path) as conn:
        conn.row_factory = sqlite3.Row
        _rollup_day_with_conn(conn, date_str)
        conn.commit()


def _rollup_day_with_conn(conn: sqlite3.Connection, date_str: str) -> None:
    conn.row_factory = sqlite3.Row
    request_rows = conn.execute(
        "SELECT endpoint, method, status, duration_ms, ts FROM request_events "
        "WHERE substr(ts,1,10) = ?",
        (date_str,),
    ).fetchall()
    why_rows = conn.execute(
        "SELECT * FROM why_events WHERE substr(ts,1,10) = ?",
        (date_str,),
    ).fetchall()
    signal_rows = conn.execute(
        "SELECT step, status, duration_ms, as_of FROM signal_events WHERE substr(ts,1,10) = ?",
        (date_str,),
    ).fetchall()
    user_rows = conn.execute(
        "SELECT event FROM user_events WHERE substr(ts,1,10) = ?",
        (date_str,),
    ).fetchall()
    health_rows = conn.execute(
        "SELECT kind FROM health_events WHERE substr(ts,1,10) = ?",
        (date_str,),
    ).fetchall()
    # narrative_cache.created_at is stored either as SQLite CURRENT_TIMESTAMP
    # ("YYYY-MM-DD HH:MM:SS") or isoformat with offset; substr(1,10) covers both.
    narrative_rows = conn.execute(
        "SELECT narrative FROM narrative_cache WHERE substr(created_at,1,10) = ?",
        (date_str,),
    ).fetchall()

    req = _agg_requests(request_rows)
    why = _agg_why(why_rows)
    feed_stats = _agg_signals(signal_rows)
    word_counts = _agg_narratives(narrative_rows)
    why_total = why["why_total"]
    cache_hits = why["cache_hits"]

    feeds: Dict[str, Any] = {}
    for step, s in sorted(feed_stats.items()):
        feeds[step] = {
            "count": s["count"],
            "present": s["present"],
            "absent": s["absent"],
            "unavailable": s["unavailable"],
            "p50_ms": _percentile(s["durations"], 0.50),
            "p95_ms": _percentile(s["durations"], 0.95),
            "max_age_hours": s["max_age_hours"],
            "as_of_count": s["as_of_count"],
        }

    payload = {
        "scale": {
            "request_count": req["request_count"],
            "why_count": why_total,
            "user_event_counts": dict(sorted(Counter(r["event"] for r in user_rows).items())),
            "distinct_locations": len(why["loc_keys"]),
        },
        "performance": {
            "per_endpoint": _latency_summary(req["latency"]),
            "overall": {
                "count": req["request_count"],
                "p50_ms": _percentile(req["all_latency"], 0.50),
                "p95_ms": _percentile(req["all_latency"], 0.95),
                "p99_ms": _percentile(req["all_latency"], 0.99),
                "max_ms": round(max(req["all_latency"]), 2) if req["all_latency"] else None,
            },
            "why_wall": {
                "avg_ms": round(sum(why["wall_ms"]) / len(why["wall_ms"]), 1) if why["wall_ms"] else None,
                "p95_ms": _percentile(why["wall_ms"], 0.95),
            },
        },
        "efficiency": {
            "llm_cost_usd": _llm_cost_total(why),
            "judge_cost_usd": why["judge_cost_usd"],
            "llm_input_tokens": why["llm_in_tokens"],
            "llm_output_tokens": why["llm_out_tokens"],
            "judge_input_tokens": why["judge_in_tokens"],
            "judge_output_tokens": why["judge_out_tokens"],
            "llm_token_rows": why["llm_token_rows"],
            "judge_token_rows": why["judge_token_rows"],
            "cache_hits": cache_hits,
            "cache_misses": why_total - cache_hits,
        },
        "quality": {
            "why_total": why_total,
            "cache_hits": cache_hits,
            "fallback_count": why["fallback_count"],
            "llm_generated": why["llm_generated"],
            "gatekeeper_retry_count": why["retry_count"],
            "judge_counts": dict(sorted(why["judge_counts"].items())),
            "top_hypotheses": dict(sorted(why["top_hypotheses"].items(), key=lambda kv: -kv[1])),
            "narrative_count": len(word_counts),
            "narrative_words_sum": int(sum(word_counts)),
            "narratives_over_150": int(sum(1 for w in word_counts if w > 150)),
        },
        "reliability": {
            "http_5xx_count": sum(n for s, n in req["status_counter"].items() if s >= 500),
            "http_503_count": req["status_counter"].get(503, 0),
            "http_429_count": req["status_counter"].get(429, 0),
            "db_write_failures": sum(1 for r in health_rows if r["kind"] == "db_write"),
            "feeds": feeds,
        },
    }
    conn.execute(
        "INSERT OR REPLACE INTO daily_rollup (date, payload_json) VALUES (?, ?)",
        (date_str, json.dumps(payload)),
    )


def backfill_rollups(days: Optional[int] = None, db_path: str = DB_PATH) -> int:
    """Roll up every completed UTC day from the earliest raw row to ``now - days``.

    ``days`` defaults to ``METRICS_RETENTION_DAYS``. Only days strictly before
    ``date(now - days)`` are rolled up, so the day the retention cutoff falls on
    is never rolled up: its raw rows stay in the active window and the hybrid
    report reads them directly (rolling that boundary day up and pruning its
    pre-cutoff sliver would lose data). Returns the number of days rolled up.
    """
    days = days or METRICS_RETENTION_DAYS
    now = datetime.now(timezone.utc)
    end = (now - timedelta(days=days)).date()
    with _db(db_path) as conn:
        earliest = None
        for table in ("request_events", "why_events", "signal_events", "user_events", "health_events"):
            try:
                d = conn.execute(f"SELECT MIN(substr(ts,1,10)) FROM {table}").fetchone()[0]
            except Exception:
                d = None
            if d and (earliest is None or d < earliest):
                earliest = d
        if earliest is None:
            return 0
        day = datetime.strptime(earliest, "%Y-%m-%d").date()
        rolled = 0
        while day < end:
            day_str = day.isoformat()
            if conn.execute("SELECT 1 FROM daily_rollup WHERE date = ?", (day_str,)).fetchone() is None:
                _rollup_day_with_conn(conn, day_str)
                rolled += 1
            day += timedelta(days=1)
        conn.commit()
    return rolled


def _report_hybrid(days: int, db_path: str, retention: int) -> Dict[str, Any]:
    """Merge daily rollups (pruned range) with raw rows (active range).

    Pruned range  ``[now-days, now-retention)`` is read from ``daily_rollup``
    (whole UTC days only); active range ``[now-retention, now)`` is read from
    raw rows aligned to the start of the retention boundary day so the two
    ranges never overlap and additive metrics merge exactly.
    """
    now = datetime.now(timezone.utc)
    full_start_date = (now - timedelta(days=days)).date().isoformat()
    active_start_date = (now - timedelta(days=retention)).date().isoformat()
    active_cutoff = f"{active_start_date}T00:00:00+00:00"

    with _db(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rollup_rows = conn.execute(
            "SELECT date, payload_json FROM daily_rollup WHERE date >= ? AND date < ?",
            (full_start_date, active_start_date),
        ).fetchall()
        raw = _fetch_raw_rows(conn, active_cutoff)
    rollups = [(r["date"], json.loads(r["payload_json"])) for r in rollup_rows]

    req = _agg_requests(raw["request_rows"])
    why = _agg_why(raw["why_rows"])
    feed_stats = _agg_signals(raw["signal_rows"])
    cached_judge_counts = _agg_verdicts(raw["verdict_rows"])
    word_counts = _agg_narratives(raw["narrative_rows"])
    raw_user_events = Counter(r["event"] for r in raw["user_rows"])
    raw_db_write_failures = sum(1 for r in raw["health_rows"] if r["kind"] == "db_write")
    cached_total = raw["cached_total"]

    # ------------------------------------------------------- additive merge
    total_requests = req["request_count"] + sum(p["scale"]["request_count"] for _, p in rollups)
    total_why = why["why_total"] + sum(p["scale"]["why_count"] for _, p in rollups)
    cache_hits = why["cache_hits"] + sum(p["efficiency"]["cache_hits"] for _, p in rollups)
    cache_misses = total_why - cache_hits
    llm_generated = why["llm_generated"] + sum(p["quality"]["llm_generated"] for _, p in rollups)
    fallback_count = why["fallback_count"] + sum(p["quality"]["fallback_count"] for _, p in rollups)
    retry_count = why["retry_count"] + sum(p["quality"]["gatekeeper_retry_count"] for _, p in rollups)
    llm_in_tokens = why["llm_in_tokens"] + sum(p["efficiency"]["llm_input_tokens"] for _, p in rollups)
    llm_out_tokens = why["llm_out_tokens"] + sum(p["efficiency"]["llm_output_tokens"] for _, p in rollups)
    judge_in_tokens = why["judge_in_tokens"] + sum(p["efficiency"]["judge_input_tokens"] for _, p in rollups)
    judge_out_tokens = why["judge_out_tokens"] + sum(p["efficiency"]["judge_output_tokens"] for _, p in rollups)
    llm_token_rows = why["llm_token_rows"] + sum(p["efficiency"]["llm_token_rows"] for _, p in rollups)
    judge_token_rows = why["judge_token_rows"] + sum(p["efficiency"]["judge_token_rows"] for _, p in rollups)
    llm_cost_total = _llm_cost_total(why) + sum(p["efficiency"]["llm_cost_usd"] for _, p in rollups)
    judge_cost_usd = why["judge_cost_usd"] + sum(p["efficiency"]["judge_cost_usd"] for _, p in rollups)
    five_xx = sum(n for s, n in req["status_counter"].items() if s >= 500) + sum(
        p["reliability"]["http_5xx_count"] for _, p in rollups
    )
    err_503 = req["status_counter"].get(503, 0) + sum(p["reliability"]["http_503_count"] for _, p in rollups)
    err_429 = req["status_counter"].get(429, 0) + sum(p["reliability"]["http_429_count"] for _, p in rollups)
    db_write_failures = raw_db_write_failures + sum(p["reliability"]["db_write_failures"] for _, p in rollups)

    judge_counts = Counter(why["judge_counts"])
    top_hypotheses = Counter(why["top_hypotheses"])
    user_events = Counter(raw_user_events)
    daily_requests = Counter(req["daily_counts"])
    daily_why = Counter(why["why_daily"])
    distinct_locations = len(why["loc_keys"])
    # Approximation: per-day distinct counts are summed, so cross-day repeats over-count.
    for date, p in rollups:
        judge_counts.update(p["quality"]["judge_counts"])
        top_hypotheses.update(p["quality"]["top_hypotheses"])
        user_events.update(p["scale"]["user_event_counts"])
        daily_requests[date] += p["scale"]["request_count"]
        daily_why[date] += p["scale"]["why_count"]
        distinct_locations += p["scale"]["distinct_locations"]
    judged = judge_counts.get("pass", 0) + judge_counts.get("fail", 0)
    cached_judged = cached_judge_counts.get("pass", 0) + cached_judge_counts.get("fail", 0)

    narrative_count = len(word_counts) + sum(p["quality"]["narrative_count"] for _, p in rollups)
    narrative_words = int(sum(word_counts)) + sum(p["quality"]["narrative_words_sum"] for _, p in rollups)
    narratives_over_150 = sum(1 for w in word_counts if w > 150) + sum(
        p["quality"]["narratives_over_150"] for _, p in rollups
    )

    # ---------------------------------------------------- latency (approx)
    endpoint_units: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for _, p in rollups:
        for endpoint, unit in (p["performance"]["per_endpoint"] or {}).items():
            endpoint_units[endpoint].append(unit)
    raw_latency_summary = _latency_summary(req["latency"])
    for endpoint, unit in raw_latency_summary.items():
        endpoint_units[endpoint].append(unit)
    merged_endpoints = {ep: _merge_latency_units(units) for ep, units in sorted(endpoint_units.items())}

    overall_units = [
        p["performance"]["overall"]
        for _, p in rollups
        if (p["performance"]["overall"] or {}).get("count")
    ]
    if req["request_count"]:
        overall_units.append({
            "count": req["request_count"],
            "p50_ms": _percentile(req["all_latency"], 0.50),
            "p95_ms": _percentile(req["all_latency"], 0.95),
            "p99_ms": _percentile(req["all_latency"], 0.99),
            "max_ms": round(max(req["all_latency"]), 2),
        })
    merged_overall = _merge_latency_units(overall_units)

    wall_units: List[Dict[str, Any]] = []
    for _, p in rollups:
        w = p["performance"]["why_wall"] or {}
        day_why = p["scale"]["why_count"]
        if day_why and (w.get("avg_ms") is not None or w.get("p95_ms") is not None):
            wall_units.append({"count": day_why, "avg_ms": w.get("avg_ms"), "p95_ms": w.get("p95_ms")})
    if why["why_total"]:
        wall_units.append({
            "count": why["why_total"],
            "avg_ms": round(sum(why["wall_ms"]) / len(why["wall_ms"]), 1) if why["wall_ms"] else None,
            "p95_ms": _percentile(why["wall_ms"], 0.95),
        })
    merged_wall = {
        "avg_ms": _weighted_mean(wall_units, "avg_ms", nd=1),
        "p95_ms": _weighted_mean(wall_units, "p95_ms"),
    }

    step_units: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for _, p in rollups:
        for step, unit in (p["reliability"]["feeds"] or {}).items():
            step_units[step].append(unit)
    for step, s in sorted(feed_stats.items()):
        step_units[step].append({
            "count": s["count"],
            "present": s["present"],
            "absent": s["absent"],
            "unavailable": s["unavailable"],
            "p50_ms": _percentile(s["durations"], 0.50),
            "p95_ms": _percentile(s["durations"], 0.95),
            "max_age_hours": s["max_age_hours"],
            "as_of_count": s["as_of_count"],
        })
    feeds: Dict[str, Any] = {}
    for step, units in sorted(step_units.items()):
        count = sum(u["count"] for u in units)
        present = sum(u["present"] for u in units)
        absent = sum(u["absent"] for u in units)
        unavailable = sum(u["unavailable"] for u in units)
        ages = [u["max_age_hours"] for u in units if u.get("max_age_hours") is not None]
        feeds[step] = {
            "count": count,
            "present": present,
            "absent": absent,
            "unavailable": unavailable,
            "availability_pct": round(100 * (present + absent) / max(1, count), 1),
            "p50_ms": _weighted_mean(units, "p50_ms"),
            "p95_ms": _weighted_mean(units, "p95_ms"),
            "max_age_hours": round(max(ages), 1) if ages else None,
            "as_of_count": sum(u.get("as_of_count", 0) for u in units),
        }

    peak_hour = max(req["hourly_counts"].items(), key=lambda kv: kv[1]) if req["hourly_counts"] else ("", 0)

    return {
        "generated_at": _utcnow_iso(),
        "window_days": days,
        "rolled_up": True,
        "rolled_up_days": len(rollups),
        "requests": {
            "total": total_requests,
            # Endpoint-status breakdown is not stored in the rollup; it reflects
            # the active raw window only.
            "by_endpoint_status": dict(sorted(req["endpoint_status_counts"].items())),
            "daily": dict(sorted(daily_requests.items())),
            "latency": merged_endpoints,
        },
        "why": {
            "total": total_why,
            "cache_hit_rate": _rate(cache_hits, total_why),
            "cache_hits": cache_hits,
            "llm_generated": llm_generated,
            "llm_cost_per_explanation_usd": round(llm_cost_total / total_why, 6) if total_why else None,
            "llm_cost_usd": round(llm_cost_total, 6),
            "judge": dict(sorted(judge_counts.items())),
            "judge_pass_rate": _rate(judge_counts["pass"], judged),
            # active raw window only
            "cached_judge": dict(sorted(cached_judge_counts.items())),
            # active raw window only
            "cached_judge_pass_rate": _rate(cached_judge_counts["pass"], cached_judged),
        },
        "narratives": {
            "count": narrative_count,
            "avg_words": round(narrative_words / narrative_count, 1) if narrative_count else None,
            # Per-day word percentiles are not stored in the rollup.
            "median_words": None,
            "p90_words": None,
            "pct_over_150_words": (
                round(100 * narratives_over_150 / narrative_count, 1) if narrative_count else None
            ),
        },
        "scale": {
            "requests_total": total_requests,
            "requests_daily": dict(sorted(daily_requests.items())),
            "why_total": total_why,
            "why_daily": dict(sorted(daily_why.items())),
            "distinct_locations": distinct_locations,
            # active raw window only
            "coverage_modes": dict(sorted(why["coverage_modes"].items())),
            "narratives_cached_total": cached_total,
            "user_events": dict(sorted(user_events.items())),
        },
        "performance": {
            "endpoints": merged_endpoints,
            "overall": merged_overall,
            # active raw window only
            "peak_req_per_hour": {"hour": peak_hour[0], "count": peak_hour[1]},
            "why_wall_ms": merged_wall,
            # active raw window only
            "cache_split_ms": {
                "cache_hit_p95_ms": _percentile(why["wall_cache_hit"], 0.95),
                "cache_miss_p95_ms": _percentile(why["wall_cache_miss"], 0.95),
            },
            "steps": feeds,
            "percentiles_note": "approximate (daily rollup)",
        },
        "quality": {
            "judge": dict(sorted(judge_counts.items())),
            "judge_pass_rate": _rate(judge_counts["pass"], judged),
            "fallback_count": fallback_count,
            "fallback_rate": _rate(fallback_count, total_why),
            "gatekeeper_retry_count": retry_count,
            "gatekeeper_retry_rate": _rate(retry_count, llm_generated),
            "top_hypotheses": dict(sorted(top_hypotheses.items(), key=lambda kv: -kv[1])),
            "narrative_length": {
                "avg_words": round(narrative_words / narrative_count, 1) if narrative_count else None,
                "pct_over_150_words": (
                    round(100 * narratives_over_150 / narrative_count, 1) if narrative_count else None
                ),
            },
            "agreement": None,
        },
        "efficiency": {
            "cache_hit_rate": _rate(cache_hits, total_why),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "llm_cost_usd": round(llm_cost_total, 6),
            "llm_cost_per_explanation_usd": round(llm_cost_total / total_why, 6) if total_why else None,
            "judge_cost_usd": round(judge_cost_usd, 6),
            "judge_cost_per_explanation_usd": round(judge_cost_usd / total_why, 6) if total_why else None,
            "avg_input_tokens": round(llm_in_tokens / llm_token_rows, 1) if llm_token_rows else None,
            "avg_output_tokens": round(llm_out_tokens / llm_token_rows, 1) if llm_token_rows else None,
            "avg_judge_tokens": (
                round((judge_in_tokens + judge_out_tokens) / judge_token_rows, 1) if judge_token_rows else None
            ),
            "cost_saved_by_cache_usd": (
                round(cache_hits * (llm_cost_total / max(1, total_why - cache_hits)), 6)
                if total_why - cache_hits else None
            ),
        },
        "reliability": {
            "feeds": feeds,
            "http": {
                "5xx_count": five_xx,
                "5xx_rate": _rate(five_xx, total_requests),
                "503_count": err_503,
                "503_rate": _rate(err_503, total_requests),
                "429_count": err_429,
                "429_rate": _rate(err_429, total_requests),
                "uptime_proxy_pct": (
                    round(100 * (total_requests - five_xx) / total_requests, 2) if total_requests else None
                ),
            },
            "health": {"db_write_failures": db_write_failures},
        },
    }


def prune(days: Optional[int] = None, db_path: str = DB_PATH) -> Dict[str, int]:
    """Delete raw event rows older than ``days`` (default METRICS_RETENTION_DAYS).

    Safe by construction: ``backfill_rollups`` runs first so every deleted row's
    date has a verified ``daily_rollup`` entry; rows whose date has no rollup
    (e.g. a backfill gap) are left untouched rather than silently lost.

    A rollup row must also be *valid* before it authorizes deletion: its
    ``payload_json`` must parse to a dict carrying all five top-level sections
    (``scale``, ``performance``, ``efficiency``, ``quality``, ``reliability``).
    Corrupt, incomplete, or placeholder payloads never authorize pruning, so
    their raw rows are left in place.

    Known residual limitation: a raw event inserted *after* its date's rollup is
    written (a "late" row for an already-rolled old date) is not captured by that
    rollup but will still be deleted by a later prune, because event rows carry
    only their logical ``ts`` (no insert time to compare against). This is not
    reachable in the live pipeline: events are timestamped at write time with the
    current UTC clock, so an old-date row can only arrive via clock skew or
    manual DB manipulation -- and the accuracy eval uses a separate database
    (``accuracy.db``), not ``cache.db``.
    """
    days = days or METRICS_RETENTION_DAYS
    backfill_rollups(days, db_path)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    cutoff_date = cutoff[:10]

    # Only a *valid* rollup payload (dict with all five required sections)
    # authorizes deletion of that date's raw rows. A corrupt, incomplete, or
    # placeholder payload leaves the raw rows untouched.
    required_sections = ("scale", "performance", "efficiency", "quality", "reliability")
    verified_dates: List[str] = []
    with _db(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute(
            "SELECT date, payload_json FROM daily_rollup WHERE date < ?",
            (cutoff_date,),
        ):
            try:
                payload = json.loads(row["payload_json"])
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict) and all(
                key in payload for key in required_sections
            ):
                verified_dates.append(row["date"])

        removed: Dict[str, int] = {}
        # SQLite caps bound variables at 999; chunk with headroom to stay safe.
        chunk_size = 900
        for table in ("request_events", "why_events", "signal_events", "user_events", "health_events"):
            total = 0
            try:
                for start in range(0, len(verified_dates), chunk_size):
                    chunk = verified_dates[start : start + chunk_size]
                    placeholders = ",".join("?" * len(chunk))
                    cur = conn.execute(
                        f"DELETE FROM {table} "
                        f"WHERE ts < ? AND substr(ts,1,10) IN ({placeholders})",
                        (cutoff, *chunk),
                    )
                    total += cur.rowcount
                removed[table] = total
            except Exception:
                removed[table] = -1
        conn.commit()
    return removed


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="backend.metrics", description=__doc__)
    parser.add_argument("--db", default=DB_PATH, help="SQLite path (default: backend/cache.db)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    report_parser = subparsers.add_parser("report", help="aggregate metrics into JSON")
    report_parser.add_argument("--days", type=int, default=30, help="lookback window in days")
    report_parser.add_argument("--out", help="optional JSON output path")

    prune_parser = subparsers.add_parser("prune", help="delete raw event rows older than --days")
    prune_parser.add_argument("--days", type=int, default=None, help=f"retention in days (default: {METRICS_RETENTION_DAYS})")

    rollup_parser = subparsers.add_parser("rollup", help="roll completed raw days into daily_rollup")
    rollup_parser.add_argument("--days", type=int, default=None, help=f"retention in days (default: {METRICS_RETENTION_DAYS})")

    args = parser.parse_args(argv)
    init_db(args.db)

    if args.command == "report":
        # Backfill missing rollups so a >retention window report self-heals (idempotent).
        if args.days > METRICS_RETENTION_DAYS:
            backfill_rollups(db_path=args.db)
        result = report(days=args.days, db_path=args.db)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2)
            print(f"wrote metrics to {args.out}")
        else:
            print(json.dumps(result, indent=2))
        return 0

    if args.command == "rollup":
        rolled = backfill_rollups(days=args.days, db_path=args.db)
        print(json.dumps({"rolled_up_days": rolled}))
        return 0

    removed = prune(days=args.days, db_path=args.db)
    print(json.dumps(removed, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
