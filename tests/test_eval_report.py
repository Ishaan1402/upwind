"""Tests for the eval dashboard renderer and its CLI surface."""

import json

from backend.eval_report import render_dashboard
from backend import eval as eval_module


SAMPLE_CORPUS = [
    {
        "scenario": "wildfire_smoke",
        "top_hypothesis": "wildfire_smoke",
        "narrative": "Smoke is drifting in from the northwest.",
        "verdict": {
            "verdict": "pass",
            "judge_model": "llama-3.3-70b-versatile",
            "reasoning": "Matches FIRMS hotspots upwind.",
        },
    }
]

SAMPLE_COMPARE = [
    {
        "scenario": "wildfire_smoke",
        "default_verdict": {"verdict": "pass"},
        "alt_verdict": {"verdict": "pass"},
    }
]

SAMPLE_STATS = {
    "total": 3,
    "verdict_counts": {"pass": 2, "fail": 1},
    "pass_rate": 0.667,
    "top_hallucinations": [["invented fire", 1]],
    "top_jargon": [["AOD", 1]],
    "by_top_hypothesis": {"wildfire_smoke": 2, "urban_pm": 1},
    "judge_models": {"llama-3.3-70b-versatile": 3},
}


def test_render_dashboard_includes_all_sections():
    page = render_dashboard(
        corpus=SAMPLE_CORPUS,
        compare=SAMPLE_COMPARE,
        stats=SAMPLE_STATS,
        rule_hits=[
            {"cache_key": "abc", "created_at": "2026-08-04", "jargon": ["AOD"], "has_disallowed_headers": False, "has_tip_heuristic": False}
        ],
        generated_at="2026-08-04 06:00 UTC",
    )
    assert "2026-08-04 06:00 UTC" in page
    assert "Corpus eval (1 fixed scenario)" in page
    assert "wildfire_smoke" in page
    assert "Judge comparison" in page
    assert "Production stats (VM narrative cache)" in page
    assert "66.7%" in page
    assert "Rule judge" in page
    assert "No rule-judge violations" not in page
    assert "abc" in page


def test_render_dashboard_escapes_user_content():
    page = render_dashboard(
        corpus=[
            {
                "scenario": "smoke",
                "top_hypothesis": "wildfire_smoke",
                "narrative": "<script>alert(1)</script> & more",
                "verdict": {"verdict": "pass", "judge_model": "m", "reasoning": "ok"},
            }
        ],
        stats={},
    )
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt; &amp; more" in page


def test_render_dashboard_empty_inputs_still_build():
    page = render_dashboard()
    assert "No corpus results yet" in page
    assert "No judge comparison results yet" in page
    assert "No judged narratives in the VM cache yet" in page
    assert "No rule-judge violations in the cache" in page


def test_cli_stats_and_rule_judge_out(tmp_path, monkeypatch):
    import sqlite3

    from backend import db as db_module

    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(db_path))
    db_module.init_db()
    db_module.set_cached_narrative("k1", "n1", {"hypotheses": [{"id": "wildfire_smoke"}]})
    db_module.update_cached_verdict("k1", {"verdict": "pass", "judge_model": "m"})
    db_module.set_cached_narrative("k2", "n2", {"hypotheses": [{"id": "urban_pm"}]})
    db_module.update_cached_verdict("k2", {"verdict": "fail", "judge_model": "m"})

    stats_out = tmp_path / "stats.json"
    rule_out = tmp_path / "rule.json"
    assert eval_module.main(["--db", str(db_path), "stats", "--out", str(stats_out)]) == 0
    assert eval_module.main(["--db", str(db_path), "rule-judge", "--out", str(rule_out)]) == 0

    stats = json.loads(stats_out.read_text())
    assert stats["total"] == 2
    assert stats["verdict_counts"] == {"pass": 1, "fail": 1}
    hits = json.loads(rule_out.read_text())
    assert isinstance(hits, list)


def test_cli_render_dashboard_end_to_end(tmp_path):
    corpus = tmp_path / "corpus.json"
    compare = tmp_path / "compare.json"
    stats = tmp_path / "stats.json"
    rule = tmp_path / "rule.json"
    out = tmp_path / "eval.html"
    corpus.write_text(json.dumps(SAMPLE_CORPUS))
    compare.write_text(json.dumps(SAMPLE_COMPARE))
    stats.write_text(json.dumps(SAMPLE_STATS))
    rule.write_text(json.dumps([]))

    assert eval_module.main(
        [
            "render-dashboard",
            "--corpus", str(corpus),
            "--compare", str(compare),
            "--stats", str(stats),
            "--rule-hits", str(rule),
            "--out", str(out),
        ]
    ) == 0
    page = out.read_text()
    assert "wildfire_smoke" in page
    assert "66.7%" in page
    assert "No rule-judge violations in the cache" in page
