"""Tests for the narrative eval tooling (backend/eval.py)."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

from backend import db as db_module
from backend import eval as eval_module
from backend import llm_judge as judge_module


def test_rule_judge_detects_jargon_units_headers():
    result = eval_module.rule_judge(
        "AOD readings and FIRMS hotspots; PM2.5 at 12 µg/m³ (90% confidence). **Briefing**"
    )
    assert "AOD" in result["jargon"]
    assert "FIRMS" in result["jargon"]
    assert "µg/m³" in result["jargon"]
    assert "90%" in result["jargon"]
    assert result["has_disallowed_headers"] is True


def test_rule_judge_clean_narrative_passes():
    result = eval_module.rule_judge(
        "Air quality is moderate today. Sensitive groups may want to limit outdoor activity."
    )
    assert result["jargon"] == []
    assert result["has_disallowed_headers"] is False
    assert result["has_tip_heuristic"] is True


def _seed_db(db_path):
    db_module.init_db()
    db_module.set_cached_narrative("k1", "n1", {"hypotheses": [{"id": "wildfire_smoke"}]})
    db_module.update_cached_verdict(
        "k1",
        {"verdict": "pass", "hallucinations": [], "leaked_jargon": [], "judge_model": "model-a"},
    )
    db_module.set_cached_narrative("k2", "n2", {"hypotheses": [{"id": "urban_pm"}]})
    db_module.update_cached_verdict(
        "k2",
        {"verdict": "fail", "hallucinations": ["invented fire"], "leaked_jargon": ["AOD"], "judge_model": "model-a"},
    )
    db_module.set_cached_narrative("k3", "n3", {"hypotheses": [{"id": "urban_pm"}]})
    db_module.update_cached_verdict(
        "k3",
        {"verdict": "pass", "hallucinations": [], "leaked_jargon": [], "judge_model": "model-a"},
    )


def test_stats_aggregates(monkeypatch, tmp_path):
    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(db_path))
    _seed_db(db_path)

    stats = eval_module.compute_stats(str(db_path))

    assert stats["total"] == 3
    assert stats["verdict_counts"] == {"pass": 2, "fail": 1}
    assert stats["pass_rate"] == round(2 / 3, 3)
    assert stats["top_hallucinations"] == [("invented fire", 1)]
    assert stats["top_jargon"] == [("AOD", 1)]
    assert stats["by_top_hypothesis"] == {"wildfire_smoke": 1, "urban_pm": 2}
    assert stats["judge_models"] == {"model-a": 3}


def test_export_fails_json(monkeypatch, tmp_path):
    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(db_path))
    _seed_db(db_path)
    out = tmp_path / "fails.json"

    records = eval_module.export_fails(str(db_path), str(out), "json")

    assert len(records) == 1
    assert records[0]["cache_key"] == "k2"
    assert json.loads(out.read_text())[0]["cache_key"] == "k2"


def test_export_fails_csv(monkeypatch, tmp_path):
    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(db_path))
    _seed_db(db_path)
    out = tmp_path / "fails.csv"

    eval_module.export_fails(str(db_path), str(out), "csv")

    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("cache_key")
    assert lines[1].startswith("k2")


def test_corpus_run_shape():
    verdict = {
        "verdict": "pass",
        "judge_model": "model-a",
        "reasoning": "ok",
        "hallucinations": [],
        "leaked_jargon": [],
    }
    with patch("backend.eval.generate_narrative_briefing", new_callable=AsyncMock, return_value="narrative"), \
         patch("backend.eval.judge_narrative", new_callable=AsyncMock, return_value=verdict):
        results = asyncio.run(eval_module.run_corpus())

    assert len(results) == 6
    assert {r["scenario"] for r in results} == {
        "wildfire_smoke", "ozone_episode", "windblown_dust",
        "winter_stagnation", "urban_pm", "good_aqi",
    }
    assert all(r["verdict"]["verdict"] == "pass" for r in results)


def test_judge_compare_reports_agreement():
    async def fake_judge(evidence, narrative):
        current = judge_module.DEFAULT_JUDGE_MODEL
        return {
            "verdict": "pass" if current == "alt-model" else "fail",
            "judge_model": current,
        }

    with patch("backend.eval.generate_narrative_briefing", new_callable=AsyncMock, return_value="n"), \
         patch("backend.eval.judge_narrative", side_effect=fake_judge):
        results = asyncio.run(eval_module.run_judge_compare("alt-model"))

    assert len(results) == 6
    assert all(r["default_verdict"]["verdict"] == "fail" for r in results)
    assert all(r["alt_verdict"]["verdict"] == "pass" for r in results)


def test_judge_verdict_includes_model():
    from types import SimpleNamespace

    content = json.dumps({
        "verdict": "pass", "grounded": True, "hallucinations": [],
        "leaked_jargon": [], "reasoning": "ok",
    })
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=AsyncMock(return_value=response))
    ))
    with patch("openai.AsyncOpenAI", return_value=client), \
         patch("backend.llm_judge.GROQ_API_KEY", "test-key"):
        verdict = asyncio.run(judge_module.judge_narrative({}, "narrative"))

    assert verdict["verdict"] == "pass"
    assert verdict["judge_model"] == judge_module.DEFAULT_JUDGE_MODEL
