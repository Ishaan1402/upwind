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
            "judge_model": "openai/gpt-oss-120b",
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
    "judge_models": {"openai/gpt-oss-120b": 3},
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


def test_render_dashboard_public_hides_narratives_and_cache_keys():
    page = render_dashboard(
        corpus=SAMPLE_CORPUS,
        rule_hits=[{"cache_key": "secret-key", "created_at": "2026-08-04", "jargon": ["AOD"], "has_disallowed_headers": False, "has_tip_heuristic": False}],
        metrics={"requests": {"total": 10, "latency": {}}, "why": {}},
        validation={"exact_agreement": 0.9, "cohens_kappa": 0.8, "judged_cases": 20, "results": []},
        public=True,
    )
    assert "Smoke is drifting in from the northwest." not in page
    assert "secret-key" not in page
    assert "hidden on public page" in page
    assert "Judge validation" in page
    assert "Scale" in page
    assert "Reliability" in page


def test_render_label_validation_shows_in_live_quality():
    page = render_dashboard(
        metrics={"quality": {"judge_pass_rate": 0.8}, "why": {}},
        label_validation={
            "label_source": "human",
            "exact_agreement": 0.91,
            "cohens_kappa": 0.83,
            "macro_f1": 0.76,
            "judged_cases": 46,
            "precision": {"pass": 0.9, "fail": 0.6},
            "recall": {"pass": 0.8, "fail": 0.7},
            "f1": {"pass": 0.85, "fail": 0.65},
            "results": [],
        },
    )
    assert "Judge agreement vs human labels" in page
    assert "91.0%" in page
    assert "0.83" in page
    assert "76.0%" in page  # macro F1 card
    assert "precision" in page
    assert "90.0%" in page  # pass precision
    assert "70.0%" in page  # fail recall
    # Deterministic validation (no human label_source) must NOT leak into the
    # live-quality agreement card (it stays in the benchmark section instead).
    page2 = render_dashboard(
        metrics={"quality": {}},
        validation={"exact_agreement": 0.7, "cohens_kappa": 0.5, "judged_cases": 7, "results": []},
    )
    assert "awaiting labeled samples" in page2
    assert "judge vs label agreement" not in page2
    assert "70.0%" in page2  # present only inside the offline benchmark section


def test_render_bakes_workflow_statuses():
    page = render_dashboard(
        workflows=[
            {"file": "ci.yml", "label": "CI", "status": "success", "run_number": 42,
             "branch": "main", "sha": "abc1234", "html_url": "https://example.com/42",
             "created_at": "2026-08-20T00:00:00Z"},
        ]
    )
    # Baked statuses ride in the page data and are rendered by renderBaked,
    # never fetched from the unauthenticated GitHub API.
    assert '"workflow_runs"' in page
    assert '"success"' in page
    assert '"run_number":42' in page
    assert "renderBaked" in page
    # Nav/content placeholders are always substituted, not leaked into HTML.
    assert "__NAV__" not in page
    assert "__CONTENT__" not in page
    assert "__GENERATED_AT__" not in page
    assert 'href="#overview"' in page
    # The client-side GitHub fallback is gone entirely; empty workflow_runs
    # keeps the "loading latest run…" placeholder instead of calling the API.
    page2 = render_dashboard()
    assert "loading latest run…" in page2
    assert "loadRun" not in page2
    assert "RUNS_URL" not in page2
    assert "fetch(" not in page2


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
