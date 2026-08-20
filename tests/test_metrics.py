import json

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
