"""Static HTML evidence page for Upwind observability.

Rendered by the nightly eval workflow and published to the VM
(https://getupwind.me/evidence/). One page, clean vertical nav on the left
separating observability topics: Scale, Performance, Quality, Efficiency,
Reliability, plus a clearly-labeled Offline benchmark section. Fully
self-contained: stdlib only, no CDN assets.

Usage (via the backend.eval CLI):
    python -m backend.eval render-dashboard \
        --corpus corpus.json --compare compare.json \
        --stats stats.json --rule-hits rule_hits.json \
        --metrics metrics.json --validation validation.json --out evidence.html
"""

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = "Ishaan1402/upwind"
WORKFLOWS = [("ci.yml", "CI"), ("deploy.yml", "Deploy"), ("eval.yml", "Nightly eval")]


def _badge(value: Optional[str]) -> str:
    cls = {
        "pass": "ok",
        "fail": "bad",
        "skipped": "warn",
        "unknown": "mute",
        "success": "ok",
        "failure": "bad",
    }.get(str(value), "mute")
    return f'<span class="badge {cls}">{html.escape(str(value or "none"))}</span>'


def _table(headers: List[str], rows: List[List[str]], empty_msg: str = "No data yet") -> str:
    if not rows:
        return f'<p class="empty">{html.escape(empty_msg)}</p>'
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _cards(items: List[tuple]) -> str:
    """A compact stat strip (key, value). Not boxy: one clean row of stats."""
    cells = "".join(
        f'<div class="stat"><span class="num">{value}</span><span class="lbl">{html.escape(label)}</span></div>'
        for value, label in items
    )
    return f'<div class="stats">{cells}</div>'


def _pct(value: Optional[float], nd: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.{nd}f}%"


def _money(value: Optional[float], nd: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"${value:.{nd}f}"


def _num(value: Optional[Any], suffix: str = "", nd: int = 0) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{nd}f}{suffix}"
    return f"{value}{suffix}"


def _corpus_table(corpus: List[Dict[str, Any]], public: bool = False) -> str:
    if not corpus:
        return '<p class="empty">No corpus results yet (first nightly run pending).</p>'
    rows = ""
    for item in corpus:
        verdict = item.get("verdict") or {}
        reasoning = (verdict.get("reasoning") or "").replace("\n", " ").strip()
        short = reasoning[:90] + ("..." if len(reasoning) > 90 else "")
        narrative = html.escape(item.get("narrative") or "")
        narrative_cell = (
            f"<details><summary>full</summary><pre>{narrative}</pre></details>"
            if not public
            else '<span class="muted">hidden on public page</span>'
        )
        rows += (
            "<tr>"
            f"<td>{html.escape(str(item.get('scenario') or ''))}</td>"
            f"<td>{_badge(verdict.get('verdict'))}</td>"
            f"<td>{html.escape(str(verdict.get('judge_model') or ''))}</td>"
            f"<td>{html.escape(str(item.get('top_hypothesis') or ''))}</td>"
            f"<td>{html.escape(short)}</td>"
            f"<td>{narrative_cell}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>scenario</th><th>verdict</th><th>judge model</th>"
        "<th>top hypothesis</th><th>reasoning</th><th></th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _compare_table(compare: List[Dict[str, Any]]) -> str:
    if not compare:
        return '<p class="empty">No judge comparison results yet.</p>'
    rows = []
    for item in compare:
        default = item.get("default_verdict") or {}
        alt = item.get("alt_verdict") or {}
        agree = (default.get("verdict") or None) == (alt.get("verdict") or None)
        rows.append([
            html.escape(str(item.get("scenario") or "")),
            _badge(default.get("verdict")),
            _badge(alt.get("verdict")),
            f'<span class="badge {"ok" if agree else "bad"}">{"yes" if agree else "no"}</span>',
        ])
    return _table(["scenario", "default", "candidate", "agree"], rows)


def _stats_section(stats: Optional[Dict[str, Any]]) -> str:
    if not stats:
        return '<p class="empty">No judged narratives in the VM cache yet.</p>'
    total = stats.get("total") or 0
    counts = stats.get("verdict_counts") or {}
    rate = stats.get("pass_rate")
    cards = _cards([
        (_num(total), "judged narratives"),
        (_pct(rate), "pass rate"),
        (_num(counts.get("fail", 0)), "failures"),
    ])
    count_rows = [[html.escape(str(k)), str(v)] for k, v in counts.items()]
    hypothesis_rows = [[html.escape(str(k)), str(v)] for k, v in (stats.get("by_top_hypothesis") or {}).items()]
    model_rows = [[html.escape(str(k)), str(v)] for k, v in (stats.get("judge_models") or {}).items()]
    halluc_rows = [[html.escape(str(k)), str(v)] for k, v in (stats.get("top_hallucinations") or [])]
    jargon_rows = [[html.escape(str(k)), str(v)] for k, v in (stats.get("top_jargon") or [])]
    return (
        cards
        + "<h3>Verdicts</h3>" + _table(["verdict", "count"], count_rows, "No verdicts yet.")
        + "<h3>By top hypothesis</h3>" + _table(["hypothesis", "count"], hypothesis_rows, "No hypotheses recorded.")
        + "<h3>Judge models</h3>" + _table(["model", "count"], model_rows, "No judge model data.")
        + "<h3>Top hallucinations</h3>" + _table(["term", "count"], halluc_rows, "None recorded.")
        + "<h3>Top leaked jargon</h3>" + _table(["term", "count"], jargon_rows, "None recorded.")
    )


def _rule_hits_table(rule_hits: List[Dict[str, Any]], public: bool = False) -> str:
    if not rule_hits:
        return '<p class="ok-line">No rule-judge violations in the cache.</p>'
    cache_key_col = "key" if not public else "hidden"
    rows = [
        [
            html.escape(str(hit.get("cache_key") or "")) if not public else "hidden",
            html.escape(str(hit.get("created_at") or "")),
            html.escape(", ".join(hit.get("jargon") or []) or "-"),
            "yes" if hit.get("has_disallowed_headers") else "no",
            "no" if hit.get("has_tip_heuristic") else "yes",
        ]
        for hit in rule_hits
    ]
    return _table([cache_key_col, "created", "jargon", "headers", "missing tip"], rows)


# --------------------------------------------------------------------------- #
# Live observability sections (all from production traffic, not benchmarks)    #
# --------------------------------------------------------------------------- #

def _scale_section(metrics: Optional[Dict[str, Any]]) -> str:
    if not metrics:
        return '<p class="empty">No metrics yet (first nightly run pending).</p>'
    scale = metrics.get("scale") or {}
    window = metrics.get("window_days", 30)
    cards = _cards([
        (_num(scale.get("requests_total", 0)), f"requests / {window}d"),
        (_num(scale.get("why_total", 0)), "why explanations"),
        (_num(scale.get("distinct_locations", 0)), "distinct locations"),
        (_num(scale.get("narratives_cached_total", 0)), "narratives cached"),
    ])
    daily = scale.get("requests_daily") or {}
    why_daily = scale.get("why_daily") or {}
    recent_days = sorted(set(list(daily.keys()) + list(why_daily.keys())))[-14:]
    daily_rows = [
        [html.escape(d), _num(daily.get(d, 0)), _num(why_daily.get(d, 0))]
        for d in reversed(recent_days)
    ]
    coverage_rows = [[html.escape(str(k)), _num(v)] for k, v in (scale.get("coverage_modes") or {}).items()]
    events_rows = [[html.escape(str(k)), _num(v)] for k, v in (scale.get("user_events") or {}).items()]
    return (
        cards
        + "<h3>Daily volume (last 14 days)</h3>"
        + _table(["date", "requests", "why"], daily_rows, "No request volume yet.")
        + "<h3>Coverage modes</h3>" + _table(["mode", "explanations"], coverage_rows, "No coverage data yet.")
        + "<h3>User events</h3>" + _table(["event", "count"], events_rows, "No user events yet.")
    )


def _performance_section(metrics: Optional[Dict[str, Any]]) -> str:
    if not metrics:
        return '<p class="empty">No metrics yet (first nightly run pending).</p>'
    perf = metrics.get("performance") or {}
    overall = perf.get("overall") or {}
    peak = perf.get("peak_req_per_hour") or {}
    wall = perf.get("why_wall_ms") or {}
    cache_split = perf.get("cache_split_ms") or {}
    cards = _cards([
        (_num(overall.get("p50_ms"), " ms"), "p50 latency"),
        (_num(overall.get("p95_ms"), " ms"), "p95 latency"),
        (_num(overall.get("p99_ms"), " ms"), "p99 latency"),
        (_num(peak.get("count", 0)), f"peak req/hour ({html.escape(str(peak.get('hour') or '?'))})"),
        (_num(wall.get("p95_ms"), " ms"), "why wall p95"),
        (_num(cache_split.get("cache_miss_p95_ms"), " ms"), "miss p95 vs hit p95 " + _num(cache_split.get("cache_hit_p95_ms"), " ms")),
    ])
    latency_rows = [
        [
            html.escape(str(endpoint)),
            _num(details.get("count")),
            _num(details.get("p50_ms")),
            _num(details.get("p95_ms")),
            _num(details.get("p99_ms")),
            _num(details.get("max_ms")),
        ]
        for endpoint, details in sorted((perf.get("endpoints") or {}).items())
    ]
    steps = perf.get("steps") or {}
    step_rows = [
        [
            html.escape(str(step)),
            _num(s.get("count")),
            _pct((s.get("availability_pct") or 0) / 100, 0) if s.get("availability_pct") is not None else "n/a",
            _num(s.get("p50_ms")),
            _num(s.get("p95_ms")),
        ]
        for step, s in sorted(steps.items())
    ]
    return (
        cards
        + "<h3>Latency by endpoint</h3>"
        + _table(["endpoint", "count", "p50 ms", "p95 ms", "p99 ms", "max ms"], latency_rows, "No request latency yet.")
        + "<h3>Evidence tool steps</h3>"
        + _table(["step", "count", "availability", "p50 ms", "p95 ms"], step_rows, "No tool-step data yet.")
    )


def _quality_section(metrics: Optional[Dict[str, Any]], label_validation: Optional[Dict[str, Any]] = None) -> str:
    if not metrics:
        return '<p class="empty">No metrics yet (first nightly run pending).</p>'
    why = metrics.get("why") or {}
    quality = metrics.get("quality") or {}
    narratives = metrics.get("narratives") or {}
    len_stats = quality.get("narrative_length") or {}
    over_150 = len_stats.get("pct_over_150_words")
    if over_150 is None:
        over_150 = narratives.get("pct_over_150_words")
    # Streaming traffic is judged asynchronously; prefer the write-back verdicts
    # (why_events) and fall back to cached narrative verdicts for legacy rows.
    judge_rate = (
        quality.get("judge_pass_rate")
        or why.get("judge_pass_rate")
        or why.get("cached_judge_pass_rate")
    )
    cards = _cards([
        (_pct(judge_rate), "judge pass rate (live)"),
        (_pct(quality.get("fallback_rate")), "deterministic fallback"),
        (_pct(quality.get("gatekeeper_retry_rate")), "gatekeeper retries"),
        (_num(len_stats.get("avg_words", narratives.get("avg_words")), " words"), "avg narrative length"),
        (_pct(over_150 / 100) if over_150 is not None else "n/a", "narratives &gt; 150 words"),
    ])
    agreement = quality.get("agreement")
    label_source = (label_validation or {}).get("label_source") or ""
    if label_validation and label_source in ("human", "agent"):
        precision = label_validation.get("precision") or {}
        recall = label_validation.get("recall") or {}
        f1 = label_validation.get("f1") or {}
        prf_rows = [
            ["pass", _pct(precision.get("pass")), _pct(recall.get("pass")), _pct(f1.get("pass"))],
            ["fail", _pct(precision.get("fail")), _pct(recall.get("fail")), _pct(f1.get("fail"))],
        ]
        agreement_note = (
            f'<p class="muted">Judge agreement vs {label_source} labels on sampled '
            "live narratives (the honest quality signal).</p>"
            + _cards([
                (_pct(label_validation.get("exact_agreement")), "judge vs label agreement"),
                (_num(label_validation.get("cohens_kappa"), nd=2), "Cohen’s kappa"),
                (_pct(label_validation.get("macro_f1")), "macro F1"),
                (_num(label_validation.get("judged_cases", 0)), "judged cases"),
            ])
            + _table(["label", "precision", "recall", "F1"], prf_rows, "No label-validation rows yet.")
        )
    elif agreement is not None:
        agreement_note = _cards([
            (_pct(agreement.get("exact_agreement")), "judge vs label agreement"),
            (_num(agreement.get("cohens_kappa"), nd=2), "Cohen’s kappa"),
            (_num(agreement.get("judged_cases", 0)), "judged cases"),
        ])
    else:
        agreement_note = '<p class="empty">Label-validation metrics are awaiting labeled samples of live narratives.</p>'
    verdict_rows = [[html.escape(str(k)), _num(v)] for k, v in (quality.get("judge") or {}).items()]
    hyp_rows = [[html.escape(str(k)), _num(v)] for k, v in (quality.get("top_hypotheses") or {}).items()]
    return (
        cards
        + agreement_note
        + "<h3>Judge verdicts (live)</h3>" + _table(["verdict", "count"], verdict_rows, "No live verdicts yet.")
        + "<h3>Top hypotheses</h3>" + _table(["hypothesis", "explanations"], hyp_rows, "No hypotheses recorded.")
    )


def _efficiency_section(metrics: Optional[Dict[str, Any]]) -> str:
    if not metrics:
        return '<p class="empty">No metrics yet (first nightly run pending).</p>'
    eff = metrics.get("efficiency") or {}
    cards = _cards([
        (_pct(eff.get("cache_hit_rate")), "cache hit rate"),
        (_money(eff.get("llm_cost_per_explanation_usd")), "LLM cost / explanation"),
        (_money(eff.get("judge_cost_per_explanation_usd")), "judge cost / explanation"),
        (_money(eff.get("cost_saved_by_cache_usd")), "cost saved by cache"),
    ])
    token_rows = [
        ["input", _num(eff.get("avg_input_tokens"))],
        ["output", _num(eff.get("avg_output_tokens"))],
        ["judge (in+out)", _num(eff.get("avg_judge_tokens"))],
    ]
    cost_rows = [
        ["LLM total", _money(eff.get("llm_cost_usd"))],
        ["Judge total", _money(eff.get("judge_cost_usd"))],
        ["Cache hits", _num(eff.get("cache_hits"))],
        ["Cache misses", _num(eff.get("cache_misses"))],
    ]
    return (
        cards
        + "<h3>Tokens per explanation</h3>" + _table(["stream", "avg tokens"], token_rows, "No token data yet.")
        + "<h3>Cost ledger</h3>" + _table(["line", "value"], cost_rows, "No cost data yet.")
    )


def _reliability_section(metrics: Optional[Dict[str, Any]]) -> str:
    if not metrics:
        return '<p class="empty">No metrics yet (first nightly run pending).</p>'
    rel = metrics.get("reliability") or {}
    http = rel.get("http") or {}
    health = rel.get("health") or {}
    cards = _cards([
        (_pct(http.get("uptime_proxy_pct") / 100, 2) if http.get("uptime_proxy_pct") is not None else "n/a", "uptime proxy"),
        (_pct(http.get("5xx_rate")), "5xx rate"),
        (_pct(http.get("503_rate")), "503 rate"),
        (_pct(http.get("429_rate")), "429 rate"),
        (_num(health.get("db_write_failures", 0)), "DB write failures"),
    ])
    feeds = rel.get("feeds") or {}
    feed_rows = [
        [
            html.escape(str(step)),
            _num(s.get("present", 0)),
            _num(s.get("absent", 0)),
            _num(s.get("unavailable", 0)),
            _pct((s.get("availability_pct") or 0) / 100, 1) if s.get("availability_pct") is not None else "n/a",
            _num(s.get("max_age_hours"), "h", 1),
        ]
        for step, s in sorted(feeds.items())
    ]
    return (
        cards
        + "<h3>Feed availability</h3>"
        + _table(["feed", "present", "absent", "unavailable", "availability", "staleness"], feed_rows, "No feed data yet.")
        + '<p class="muted">availability = feed responded (present or absent) vs failed/unreachable; '
        'staleness = max age of feed-reported data (as_of) in hours.</p>'
    )


def _benchmark_section(
    corpus: List[Dict[str, Any]],
    compare: List[Dict[str, Any]],
    stats: Optional[Dict[str, Any]],
    rule_hits: List[Dict[str, Any]],
    validation: Optional[Dict[str, Any]],
    public: bool,
) -> str:
    """Offline, research-grade benchmark — explicitly not live truth."""
    header = (
        '<div class="bench-note">'
        "<strong>Offline benchmark (research).</strong> "
        "These numbers come from fixed scenarios, the IMPROVE-derived answer key, "
        "and deterministic rule checks — not from live traffic. Treat them as a "
        "lab signal; the Scale/Performance/Quality/Efficiency/Reliability sections "
        "above are the live, production-measured truth."
        "</div>"
    )
    validation_html = _validation_section(validation)
    corpus_title = (
        f"Corpus eval ({len(corpus)} fixed {'scenario' if len(corpus) == 1 else 'scenarios'})"
    )
    return (
        header
        + "<h3>Judge validation (known labels)</h3>" + validation_html
        + f"<h3>{html.escape(corpus_title)}</h3>" + _corpus_table(corpus, public=public)
        + "<h3>Judge comparison</h3>" + _compare_table(compare)
        + "<h3>Production stats (VM narrative cache)</h3>" + _stats_section(stats)
        + "<h3>Rule judge</h3>" + _rule_hits_table(rule_hits, public=public)
    )


def _validation_section(validation: Optional[Dict[str, Any]]) -> str:
    if not validation:
        return '<p class="empty">No human/known-label validation results yet.</p>'
    agreement = validation.get("exact_agreement")
    kappa = validation.get("cohens_kappa")
    cards = _cards([
        (_num(validation.get("judged_cases", 0)), "judged cases"),
        (_pct(agreement), "judge agreement"),
        (_num(kappa), "Cohen’s kappa"),
    ])
    rows = []
    for item in validation.get("results") or []:
        rows.append([
            html.escape(str(item.get("name") or "")),
            _badge(item.get("gold_verdict")),
            _badge(item.get("judge_verdict")),
            "yes" if item.get("agreement") else "no",
            html.escape(str(item.get("judge_model") or "")),
        ])
    return cards + _table(
        ["case", "gold", "judge", "agree", "judge model"],
        rows,
        "No validation rows.",
    )


def render_dashboard(
    corpus: Optional[List[Dict[str, Any]]] = None,
    compare: Optional[List[Dict[str, Any]]] = None,
    stats: Optional[Dict[str, Any]] = None,
    rule_hits: Optional[List[Dict[str, Any]]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    validation: Optional[Dict[str, Any]] = None,
    label_validation: Optional[Dict[str, Any]] = None,
    workflows: Optional[List[Dict[str, Any]]] = None,
    public: bool = False,
    generated_at: Optional[str] = None,
) -> str:
    """Build the full evidence page as a single self-contained HTML string.

    ``workflows`` is the output of `python -m backend.eval workflow-status`
    (baked server-side so the page never depends on the unauthenticated
    GitHub API from visitors' browsers). ``label_validation`` is the
    human/agent-labeled agreement output from backend.human_labels validate.
    """
    if generated_at is None:
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    corpus = corpus or []
    data = json.dumps(
        {
            "generated_at": generated_at,
            "repo": REPO,
            "workflows": WORKFLOWS,
            "workflow_runs": workflows or [],
            "public": public,
        },
        separators=(",", ":"),
    ).replace("<", "\\u003c")
    workflow_cards = "".join(
        f'<div class="card wf" id="wf-{file.replace(".", "-")}">'
        f'<span class="dot mute"></span><strong>{label}</strong>'
        '<div class="wf-meta">loading latest run…</div></div>'
        for file, label in WORKFLOWS
    )
    sections = {
        "overview": (
            "<h2>Overview</h2>"
            "<p class='muted'>CI status, headline numbers, and how this page is produced.</p>"
            '<div class="cards" id="wf-cards">' + workflow_cards + "</div>"
        ),
        "scale": "<h2>Scale</h2>" + _scale_section(metrics),
        "performance": "<h2>Performance</h2>" + _performance_section(metrics),
        "quality": "<h2>Quality</h2>" + _quality_section(metrics, label_validation),
        "efficiency": "<h2>Efficiency</h2>" + _efficiency_section(metrics),
        "reliability": "<h2>Reliability</h2>" + _reliability_section(metrics),
        "benchmark": "<h2>Offline benchmark</h2>" + _benchmark_section(corpus, compare, stats, rule_hits or [], validation, public),
    }
    nav = "".join(
        f'<a href="#{sid}">{label}</a>'
        for sid, label in [
            ("overview", "Overview"),
            ("scale", "Scale"),
            ("performance", "Performance"),
            ("quality", "Quality"),
            ("efficiency", "Efficiency"),
            ("reliability", "Reliability"),
            ("benchmark", "Offline benchmark"),
        ]
    )
    content = "".join(f'<section id="{sid}">{body}</section>' for sid, body in sections.items())
    return TEMPLATE.replace("__NAV__", nav).replace("__CONTENT__", content).replace(
        "__DATA_JSON__", data
    ).replace("__GENERATED_AT__", html.escape(generated_at))


# The page markup lives in backend/eval/templates/evidence.html (stdlib-only load
# at module import, no jinja). Everything else is substituted at render time.
TEMPLATE = (Path(__file__).parent / "eval" / "templates" / "evidence.html").read_text(
    encoding="utf-8"
)
