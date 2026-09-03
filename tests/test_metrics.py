import json
import sqlite3

import pytest

from backend import db as db_module
from backend import metrics as metrics_module


def test_record_and_report(tmp_path, monkeypatch):
    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(db_path))
    metrics_module.init_db()

    metrics_module.record_request("GET", "/api/aqi", 200, 80.0, db_path=str(db_path))
    metrics_module.record_request("GET", "/api/aqi", 200, 120.0, db_path=str(db_path))
    metrics_module.record_request("GET", "/api/aqi", 200, 300.0, db_path=str(db_path))
    metrics_module.record_why("/api/why", cache_hit=False, llm_generated=True, llm_cost_usd=0.004, judge_verdict="pass", db_path=str(db_path))
    metrics_module.record_why("/api/why", cache_hit=True, llm_generated=False, llm_cost_usd=0.0, judge_verdict=None, db_path=str(db_path))

    result = metrics_module.report(days=30, db_path=str(db_path))

    assert result["requests"]["total"] == 3
    assert result["requests"]["latency"]["/api/aqi GET"]["p50_ms"] == 120.0
    assert result["requests"]["latency"]["/api/aqi GET"]["p95_ms"] == 300.0
    assert result["why"]["total"] == 2
    assert result["why"]["cache_hit_rate"] == 0.5
    assert result["why"]["llm_cost_per_explanation_usd"] == 0.002
    assert result["why"]["judge_pass_rate"] == 1.0


def test_report_narrative_verbosity(tmp_path, monkeypatch):
    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(db_path))
    metrics_module.init_db()

    db_module.set_cached_narrative("why_a", " ".join(["word"] * 80), {})   # 80 words
    db_module.set_cached_narrative("why_b", " ".join(["word"] * 140), {})  # 140 words
    db_module.set_cached_narrative("why_c", " ".join(["word"] * 200), {})  # 200 words (> 150)

    n = metrics_module.report(days=30, db_path=str(db_path))["narratives"]
    assert n["count"] == 3
    assert n["avg_words"] == 140.0
    assert n["median_words"] == 140.0
    assert n["p90_words"] == 200.0
    assert n["pct_over_150_words"] == round(100.0 / 3, 1)


def test_report_narrative_verbosity_empty(tmp_path, monkeypatch):
    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(db_path))
    metrics_module.init_db()

    n = metrics_module.report(days=30, db_path=str(db_path))["narratives"]
    assert n["count"] == 0
    assert n["avg_words"] is None
    assert n["median_words"] is None
    assert n["pct_over_150_words"] is None


def test_record_signal_events_and_feed_availability(tmp_path, monkeypatch):
    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(db_path))
    metrics_module.init_db()

    metrics_module.record_signal_events("/api/why", [
        {"step": "weather_vector", "status": "done", "duration_ms": 12.0},
        {"step": "firms_scan", "status": "warning", "duration_ms": 800.0},
        {"step": "web_search", "status": "absent", "duration_ms": 3.0},
    ], db_path=str(db_path))

    feeds = metrics_module.report(days=30, db_path=str(db_path))["performance"]["steps"]
    assert feeds["weather_vector"]["present"] == 1
    assert feeds["weather_vector"]["availability_pct"] == 100.0
    assert feeds["firms_scan"]["unavailable"] == 1
    assert feeds["firms_scan"]["availability_pct"] == 0.0
    assert feeds["web_search"]["absent"] == 1
    assert feeds["web_search"]["p50_ms"] == 3.0


def test_record_user_event_and_scale(tmp_path, monkeypatch):
    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(db_path))
    metrics_module.init_db()

    metrics_module.record_user_event("aqi_view", "90210", db_path=str(db_path))
    metrics_module.record_user_event("why_open", None, db_path=str(db_path))

    scale = metrics_module.report(days=30, db_path=str(db_path))["scale"]
    assert scale["user_events"] == {"aqi_view": 1, "why_open": 1}


def test_report_feed_staleness_max_age_hours(tmp_path, monkeypatch):
    from datetime import datetime, timezone, timedelta
    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(db_path))
    metrics_module.init_db()

    # One step reports a known as_of (2h ago); the other reports none.
    as_of_2h = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    metrics_module.record_signal_events("/api/why", [
        {"step": "openaq_monitors", "status": "done", "duration_ms": 100.0, "as_of": as_of_2h},
        {"step": "weather_vector", "status": "done", "duration_ms": 50.0},  # no as_of
        {"step": "hms_scan", "status": "done", "duration_ms": 30.0, "as_of": "not-a-datetime"},  # bad as_of tolerated
    ], db_path=str(db_path))

    feeds = metrics_module.report(days=30, db_path=str(db_path))["reliability"]["feeds"]

    # Expected age is derived from the as_of we inserted, so the assertion is
    # immune to wall-clock drift between the insert and the report call.
    expected_age = round(
        (datetime.now(timezone.utc) - datetime.fromisoformat(as_of_2h)).total_seconds() / 3600, 1
    )
    assert feeds["openaq_monitors"]["max_age_hours"] == pytest.approx(expected_age, abs=0.1)
    assert feeds["openaq_monitors"]["max_age_hours"] == pytest.approx(2.0, abs=0.1)
    assert feeds["openaq_monitors"]["as_of_count"] == 1
    assert feeds["weather_vector"]["max_age_hours"] is None
    assert feeds["weather_vector"]["as_of_count"] == 0
    # Bad as_of values are tolerated and never count toward the staleness stats.
    assert feeds["hms_scan"]["max_age_hours"] is None
    assert feeds["hms_scan"]["as_of_count"] == 0


def test_record_why_tokens_and_efficiency(tmp_path, monkeypatch):
    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(db_path))
    metrics_module.init_db()

    metrics_module.record_why(
        "/api/why",
        cache_hit=False,
        llm_generated=True,
        llm_cost_usd=0.004,
        judge_verdict="pass",
        country_code="us",
        loc_key="90210",
        total_ms=2500.0,
        top_hypothesis="wildfire_smoke",
        top_confidence="high",
        llm_input_tokens=1200,
        llm_output_tokens=180,
        judge_input_tokens=3000,
        judge_output_tokens=120,
        fallback_used=False,
        gatekeeper_retries=0,
        db_path=str(db_path),
    )
    metrics_module.record_why("/api/why", cache_hit=True, llm_generated=False, llm_cost_usd=0.0,
                              country_code="ca", loc_key="vancouver", total_ms=120.0,
                              top_hypothesis="wildfire_smoke", top_confidence="high",
                              db_path=str(db_path))

    r = metrics_module.report(days=30, db_path=str(db_path))
    assert r["scale"]["distinct_locations"] == 2
    assert r["scale"]["coverage_modes"] == {"us": 1, "ca": 1}
    assert r["quality"]["top_hypotheses"] == {"wildfire_smoke": 2}
    assert r["efficiency"]["avg_input_tokens"] == 1200.0
    assert r["efficiency"]["llm_cost_usd"] > 0
    assert r["efficiency"]["judge_cost_usd"] > 0
    assert r["efficiency"]["cache_hit_rate"] == 0.5
    assert r["reliability"]["health"]["db_write_failures"] == 0


def test_record_why_returns_id_and_update_why_verdict(tmp_path, monkeypatch):
    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(db_path))
    metrics_module.init_db()

    row_id = metrics_module.record_why("/api/why/stream", llm_generated=True, db_path=str(db_path))
    assert row_id is not None

    # Simulate the async stream judge writing its result back to the row.
    metrics_module.update_why_verdict(row_id, "pass", judge_input_tokens=3000, judge_output_tokens=120, db_path=str(db_path))

    q = metrics_module.report(days=30, db_path=str(db_path))["quality"]
    assert q["judge"] == {"pass": 1}
    assert q["judge_pass_rate"] == 1.0

    # update on a nonexistent id must be a harmless no-op.
    metrics_module.update_why_verdict(999999, "fail", db_path=str(db_path))


def test_prune_removes_old_rows(tmp_path, monkeypatch):
    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(db_path))
    metrics_module.init_db()

    # Old row inserted with a ts far in the past via direct SQL.
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO request_events (ts, endpoint, method, status, duration_ms) "
        "VALUES ('2020-01-01T00:00:00+00:00', '/api/aqi', 'GET', 200, 10.0)"
    )
    conn.commit()
    conn.close()

    removed = metrics_module.prune(days=90, db_path=str(db_path))
    assert removed["request_events"] == 1
    assert metrics_module.report(days=30, db_path=str(db_path))["requests"]["total"] == 0


def test_estimate_llm_cost_uses_output_price(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics_module, "LLM_OUTPUT_PRICE_PER_1M", 1.0)
    # 4000 chars ≈ 1000 tokens at $1/M tokens.
    assert metrics_module.estimate_llm_cost("x" * 4000) == 0.001


def test_cli_report_writes_json(tmp_path, monkeypatch):
    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(db_path))
    out = tmp_path / "metrics.json"
    assert metrics_module.main(["--db", str(db_path), "report", "--days", "7", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["window_days"] == 7


def test_rollup_day_counts_and_sums_are_deterministic(tmp_path, monkeypatch):
    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(db_path))
    metrics_module.init_db()
    day = "2026-07-15"
    ts = f"{day}T12:00:00+00:00"

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO request_events (ts, endpoint, method, status, duration_ms) "
        "VALUES (?, '/api/aqi', 'GET', 200, 80.0)", (ts,))
    conn.execute(
        "INSERT INTO request_events (ts, endpoint, method, status, duration_ms) "
        "VALUES (?, '/api/aqi', 'GET', 500, 300.0)", (ts,))
    conn.execute(
        "INSERT INTO why_events (ts, endpoint, cache_hit, llm_generated, llm_cost_usd, judge_verdict, "
        "loc_key, total_ms, top_hypothesis, llm_input_tokens, llm_output_tokens, judge_input_tokens, "
        "judge_output_tokens, fallback_used, gatekeeper_retries) "
        "VALUES (?, '/api/why', 1, 1, 0.004, 'pass', '90210', 2500.0, 'wildfire_smoke', 1200, 180, 3000, 120, 0, 1)",
        (ts,))
    conn.execute(
        "INSERT INTO signal_events (ts, endpoint, step, status, duration_ms) "
        "VALUES (?, '/api/why', 'weather_vector', 'done', 12.0)", (ts,))
    conn.execute(
        "INSERT INTO user_events (ts, event, detail) VALUES (?, 'aqi_view', '90210')", (ts,))
    conn.execute(
        "INSERT INTO health_events (ts, kind, detail) VALUES (?, 'db_write', 'x')", (ts,))
    conn.execute(
        "INSERT INTO narrative_cache (cache_key, narrative, payload_json, created_at) "
        "VALUES ('k1', ?, '{}', ?)", (" ".join(["word"] * 80), f"{day} 10:00:00"))
    conn.commit()
    conn.close()

    metrics_module.rollup_day(day, db_path=str(db_path))

    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT payload_json FROM daily_rollup WHERE date = ?", (day,)).fetchone()
    conn.close()
    assert row is not None
    payload = json.loads(row[0])

    assert payload["scale"]["request_count"] == 2
    assert payload["scale"]["why_count"] == 1
    assert payload["scale"]["user_event_counts"] == {"aqi_view": 1}
    assert payload["scale"]["distinct_locations"] == 1
    assert payload["performance"]["per_endpoint"]["/api/aqi GET"]["count"] == 2
    assert payload["performance"]["per_endpoint"]["/api/aqi GET"]["p50_ms"] == 80.0
    assert payload["performance"]["overall"]["count"] == 2
    assert payload["performance"]["why_wall"]["p95_ms"] == 2500.0
    assert payload["efficiency"]["llm_input_tokens"] == 1200
    assert payload["efficiency"]["llm_output_tokens"] == 180
    assert payload["efficiency"]["llm_token_rows"] == 1
    assert payload["efficiency"]["judge_token_rows"] == 1
    assert payload["efficiency"]["cache_hits"] == 1
    assert payload["efficiency"]["cache_misses"] == 0
    assert payload["efficiency"]["llm_cost_usd"] > 0
    assert payload["efficiency"]["judge_cost_usd"] > 0
    assert payload["quality"]["judge_counts"] == {"pass": 1}
    assert payload["quality"]["top_hypotheses"] == {"wildfire_smoke": 1}
    assert payload["quality"]["gatekeeper_retry_count"] == 1
    assert payload["quality"]["narrative_count"] == 1
    assert payload["quality"]["narrative_words_sum"] == 80
    assert payload["quality"]["narratives_over_150"] == 0
    assert payload["reliability"]["http_5xx_count"] == 1
    assert payload["reliability"]["db_write_failures"] == 1
    assert payload["reliability"]["feeds"]["weather_vector"]["present"] == 1

    # Determinism: rolling the same day again yields an identical payload.
    metrics_module.rollup_day(day, db_path=str(db_path))
    conn = sqlite3.connect(str(db_path))
    row2 = conn.execute("SELECT payload_json FROM daily_rollup WHERE date = ?", (day,)).fetchone()
    conn.close()
    assert json.loads(row2[0]) == payload


def test_prune_preserves_unverified_rows_and_deletes_verified(tmp_path, monkeypatch):
    from datetime import datetime, timezone, timedelta
    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(db_path))
    metrics_module.init_db()

    now = datetime.now(timezone.utc)
    cutoff_dt = now - timedelta(days=90)
    # The day the retention cutoff falls on is NOT rolled up by backfill_rollups
    # (its raw rows stay in the active window) -> an "unverified" gap.
    boundary_date = cutoff_dt.date().isoformat()
    verified_date = (cutoff_dt.date() - timedelta(days=1)).isoformat()
    boundary_ts = f"{boundary_date}T00:00:00+00:00"   # older than retention, no rollup
    verified_ts = f"{verified_date}T23:59:00+00:00"   # older than retention, rollup will exist

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO request_events (ts, endpoint, method, status, duration_ms) "
        "VALUES (?, '/api/aqi', 'GET', 200, 10.0)", (boundary_ts,))
    conn.execute(
        "INSERT INTO request_events (ts, endpoint, method, status, duration_ms) "
        "VALUES (?, '/api/aqi', 'GET', 200, 20.0)", (verified_ts,))
    conn.commit()
    conn.close()

    removed = metrics_module.prune(days=90, db_path=str(db_path))

    # The verified date's row is pruned; the unverified boundary-day row is kept.
    assert removed["request_events"] == 1
    conn = sqlite3.connect(str(db_path))
    remaining = conn.execute("SELECT COUNT(*) FROM request_events").fetchone()[0]
    kept = conn.execute(
        "SELECT COUNT(*) FROM request_events WHERE substr(ts,1,10) = ?", (boundary_date,)).fetchone()[0]
    conn.close()
    assert remaining == 1
    assert kept == 1


def test_prune_requires_valid_rollup_payload(tmp_path, monkeypatch):
    from datetime import datetime, timezone, timedelta
    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(db_path))
    metrics_module.init_db()

    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=90)
    # Dates well before the retention cutoff (backfill_rollups would normally cover them).
    corrupt_date = (cutoff_dt.date() - timedelta(days=400)).isoformat()
    incomplete_date = (cutoff_dt.date() - timedelta(days=401)).isoformat()
    valid_date = (cutoff_dt.date() - timedelta(days=402)).isoformat()

    conn = sqlite3.connect(str(db_path))
    for d in (corrupt_date, incomplete_date, valid_date):
        conn.execute(
            "INSERT INTO request_events (ts, endpoint, method, status, duration_ms) "
            "VALUES (?, '/api/aqi', 'GET', 200, 10.0)", (f"{d}T12:00:00+00:00",))
    # Pre-write daily_rollup rows; backfill_rollups only fills *missing* days, so
    # these survive unchanged. A corrupt or incomplete payload must NOT authorize
    # deletion of that date's raw rows; a complete five-section payload must.
    conn.execute(
        "INSERT INTO daily_rollup (date, payload_json) VALUES (?, ?)",
        (corrupt_date, "not json"))
    conn.execute(
        "INSERT INTO daily_rollup (date, payload_json) VALUES (?, ?)",
        (incomplete_date, json.dumps(
            {"scale": {}, "performance": {}, "efficiency": {}, "quality": {}})))
    conn.execute(
        "INSERT INTO daily_rollup (date, payload_json) VALUES (?, ?)",
        (valid_date, json.dumps(
            {"scale": {}, "performance": {}, "efficiency": {},
             "quality": {}, "reliability": {}})))
    conn.commit()
    conn.close()

    removed = metrics_module.prune(days=90, db_path=str(db_path))

    # Only the date whose rollup payload is valid and complete is pruned.
    assert removed["request_events"] == 1
    conn = sqlite3.connect(str(db_path))
    remaining_dates = sorted(
        r[0] for r in conn.execute(
            "SELECT DISTINCT substr(ts,1,10) FROM request_events").fetchall())
    conn.close()
    assert remaining_dates == sorted([corrupt_date, incomplete_date])


def test_report_hybrid_beyond_retention_merges_exactly(tmp_path, monkeypatch):
    from datetime import datetime, timezone, timedelta
    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(db_path))
    monkeypatch.setattr(metrics_module, "METRICS_RETENTION_DAYS", 7)
    metrics_module.init_db()

    now = datetime.now(timezone.utc)
    old_date = (now - timedelta(days=30)).date().isoformat()
    old_ts = f"{old_date}T12:00:00+00:00"
    recent_ts = (now - timedelta(days=1)).isoformat(timespec="seconds")
    recent_date = (now - timedelta(days=1)).date().isoformat()

    conn = sqlite3.connect(str(db_path))
    # Pruned-range rows: survive only via the daily rollup.
    conn.execute(
        "INSERT INTO request_events (ts, endpoint, method, status, duration_ms) "
        "VALUES (?, '/api/aqi', 'GET', 200, 100.0)", (old_ts,))
    conn.execute(
        "INSERT INTO why_events (ts, endpoint, cache_hit, llm_generated, llm_cost_usd, loc_key) "
        "VALUES (?, '/api/why', 0, 1, 0.004, '11111')", (old_ts,))
    # Active-range rows: still raw.
    conn.execute(
        "INSERT INTO request_events (ts, endpoint, method, status, duration_ms) "
        "VALUES (?, '/api/aqi', 'GET', 200, 200.0)", (recent_ts,))
    conn.execute(
        "INSERT INTO why_events (ts, endpoint, cache_hit, llm_generated, llm_cost_usd, loc_key) "
        "VALUES (?, '/api/why', 1, 0, 0.0, '22222')", (recent_ts,))
    conn.execute(
        "INSERT INTO user_events (ts, event, detail) VALUES (?, 'aqi_view', NULL)", (recent_ts,))
    conn.commit()
    conn.close()

    metrics_module.rollup_day(old_date, db_path=str(db_path))

    result = metrics_module.report(days=60, db_path=str(db_path))

    assert result["rolled_up"] is True
    assert result["rolled_up_days"] == 1
    assert result["performance"]["percentiles_note"] == "approximate (daily rollup)"
    assert result["requests"]["total"] == 2
    assert result["scale"]["requests_total"] == 2
    assert result["why"]["total"] == 2
    assert result["efficiency"]["cache_hits"] == 1
    assert result["efficiency"]["cache_misses"] == 1
    assert result["scale"]["distinct_locations"] == 2
    assert result["scale"]["user_events"] == {"aqi_view": 1}
    assert result["scale"]["requests_daily"] == {old_date: 1, recent_date: 1}
    assert result["requests"]["daily"] == result["scale"]["requests_daily"]


def test_cli_rollup_runs_backfill(tmp_path, monkeypatch):
    from datetime import datetime, timezone, timedelta
    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(db_path))
    metrics_module.init_db()

    old_ts = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat(timespec="seconds")
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO request_events (ts, endpoint, method, status, duration_ms) "
        "VALUES (?, '/api/aqi', 'GET', 200, 10.0)", (old_ts,))
    conn.commit()
    conn.close()

    assert metrics_module.main(["--db", str(db_path), "rollup"]) == 0
    conn = sqlite3.connect(str(db_path))
    n = conn.execute("SELECT COUNT(*) FROM daily_rollup").fetchone()[0]
    conn.close()
    assert n >= 1
