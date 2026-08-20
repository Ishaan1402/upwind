"""Static HTML dashboard for Upwind CI and narrative eval results.

Rendered by the nightly eval workflow and published to the VM behind nginx
basic auth (https://getupwind.me/eval/). Fully self-contained: stdlib only,
no CDN assets, so nothing about the page leaks outside the VM.

Usage (via the backend.eval CLI):
    python -m backend.eval render-dashboard \
        --corpus corpus.json --compare compare.json \
        --stats stats.json --rule-hits rule_hits.json --out eval.html
"""

import html
import json
from datetime import datetime, timezone
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
    rate_str = f"{rate * 100:.1f}%" if rate is not None else "n/a"
    cards = (
        '<div class="cards">'
        f'<div class="card"><span class="num">{total}</span>judged narratives</div>'
        f'<div class="card"><span class="num">{rate_str}</span>pass rate</div>'
        f'<div class="card"><span class="num">{counts.get("fail", 0)}</span>failures</div>'
        "</div>"
    )
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


def _metrics_section(metrics: Optional[Dict[str, Any]]) -> str:
    if not metrics:
        return '<p class="empty">No metrics yet (first nightly run pending).</p>'
    requests = metrics.get("requests") or {}
    why = metrics.get("why") or {}
    narratives = metrics.get("narratives") or {}
    latency = requests.get("latency") or {}
    rate = why.get("cache_hit_rate")
    cost = why.get("llm_cost_per_explanation_usd")
    judge_rate = why.get("judge_pass_rate") or why.get("cached_judge_pass_rate")
    rate_str = f"{rate * 100:.1f}%" if rate is not None else "n/a"
    cost_str = f"${cost:.4f}" if cost is not None else "n/a"
    judge_str = f"{judge_rate * 100:.1f}%" if judge_rate is not None else "n/a"
    avg_words = narratives.get("avg_words")
    over_150 = narratives.get("pct_over_150_words")
    avg_words_str = f"{avg_words:.0f}" if avg_words is not None else "n/a"
    over_150_str = f"{over_150:.0f}%" if over_150 is not None else "n/a"
    cards = (
        '<div class="cards">'
        f'<div class="card"><span class="num">{requests.get("total", 0)}</span>requests / {metrics.get("window_days", 30)}d</div>'
        f'<div class="card"><span class="num">{rate_str}</span>cache hit rate</div>'
        f'<div class="card"><span class="num">{cost_str}</span>LLM cost / explanation</div>'
        f'<div class="card"><span class="num">{judge_str}</span>judge pass rate</div>'
        "</div>"
        '<div class="cards">'
        f'<div class="card"><span class="num">{avg_words_str}</span>avg narrative words</div>'
        f'<div class="card"><span class="num">{over_150_str}</span>narratives &gt; 150 words</div>'
        "</div>"
    )
    latency_rows = [
        [
            html.escape(str(endpoint)),
            str(details.get("count")),
            str(details.get("p50_ms")),
            str(details.get("p95_ms")),
            str(details.get("max_ms")),
        ]
        for endpoint, details in sorted(latency.items())
    ]
    return cards + "<h3>Latency by endpoint</h3>" + _table(
        ["endpoint", "count", "p50 ms", "p95 ms", "max ms"],
        latency_rows,
        "No request latency data yet.",
    )


def _validation_section(validation: Optional[Dict[str, Any]]) -> str:
    if not validation:
        return '<p class="empty">No human/known-label validation results yet.</p>'
    agreement = validation.get("exact_agreement")
    kappa = validation.get("cohens_kappa")
    agreement_str = f"{agreement * 100:.1f}%" if agreement is not None else "n/a"
    kappa_str = str(kappa) if kappa is not None else "n/a"
    cards = (
        '<div class="cards">'
        f'<div class="card"><span class="num">{validation.get("judged_cases", 0)}</span>judged cases</div>'
        f'<div class="card"><span class="num">{agreement_str}</span>judge agreement</div>'
        f'<div class="card"><span class="num">{kappa_str}</span>Cohen’s kappa</div>'
        "</div>"
    )
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
    public: bool = False,
    generated_at: Optional[str] = None,
) -> str:
    """Build the full dashboard page as a single self-contained HTML string."""
    if generated_at is None:
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    corpus = corpus or []
    corpus_title = (
        f"Corpus eval ({len(corpus)} fixed "
        f"{'scenario' if len(corpus) == 1 else 'scenarios'})"
    )
    data = json.dumps(
        {"generated_at": generated_at, "repo": REPO, "workflows": WORKFLOWS, "public": public},
        separators=(",", ":"),
    )
    workflow_cards = "".join(
        f'<div class="card wf" id="wf-{file.replace(".", "-")}">'
        f'<span class="dot mute"></span><strong>{label}</strong>'
        '<div class="wf-meta">loading latest run…</div></div>'
        for file, label in WORKFLOWS
    )
    return TEMPLATE.replace("__DATA_JSON__", data).replace(
        "__WORKFLOW_CARDS__", workflow_cards
    ).replace(
        "__GENERATED_AT__", html.escape(generated_at)
    ).replace(
        "__CORPUS_TITLE__", corpus_title
    ).replace(
        "__CORPUS__", _corpus_table(corpus, public=public)
    ).replace(
        "__COMPARE__", _compare_table(compare or [])
    ).replace(
        "__STATS__", _stats_section(stats)
    ).replace(
        "__RULE_HITS__", _rule_hits_table(rule_hits or [], public=public)
    ).replace(
        "__METRICS__", _metrics_section(metrics)
    ).replace(
        "__VALIDATION__", _validation_section(validation)
    )


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Upwind eval dashboard</title>
<style>
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; background: #0d1117; color: #e6edf3; font: 14px/1.5 system-ui, -apple-system, sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 24px 18px 56px; }
header h1 { margin: 0 0 4px; font-size: 22px; }
header p { margin: 0; color: #8b949e; }
h2 { margin: 36px 0 10px; font-size: 17px; border-bottom: 1px solid #21262d; padding-bottom: 8px; }
h3 { margin: 22px 0 8px; font-size: 14px; color: #c9d1d9; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 14px 0; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; }
.card .num { display: block; font-size: 24px; font-weight: 700; }
.wf { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.wf strong { min-width: 90px; }
.wf-meta { color: #8b949e; font-size: 12px; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.dot.ok { background: #3fb950; }
.dot.bad { background: #f85149; }
.dot.warn { background: #d29922; }
.dot.mute { background: #484f58; }
table { width: 100%; border-collapse: collapse; margin: 8px 0; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid #21262d; vertical-align: top; }
th { color: #8b949e; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
td { font-size: 13px; }
pre { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 10px; overflow: auto; white-space: pre-wrap; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; }
.badge.ok { background: rgba(63,185,80,.15); color: #3fb950; }
.badge.bad { background: rgba(248,81,73,.15); color: #f85149; }
.badge.warn { background: rgba(210,153,34,.15); color: #d29922; }
.badge.mute { background: rgba(139,148,158,.15); color: #8b949e; }
.empty { color: #8b949e; }
.ok-line { color: #3fb950; }
details summary { cursor: pointer; color: #58a6ff; }
a { color: #58a6ff; text-decoration: none; }
a:hover { text-decoration: underline; }
footer { margin-top: 48px; color: #484f58; font-size: 12px; }
</style>
</head>
<body>
<main>
<header>
  <h1>Upwind eval dashboard</h1>
  <p>CI status, nightly corpus eval, and production judge stats.
     Updated <strong>__GENERATED_AT__</strong>.</p>
</header>

<h2>CI status</h2>
<div class="cards" id="wf-cards">__WORKFLOW_CARDS__</div>

<h2>Metrics</h2>
__METRICS__

<h2>Judge validation</h2>
__VALIDATION__

<h2>__CORPUS_TITLE__</h2>
__CORPUS__

<h2>Judge comparison</h2>
__COMPARE__

<h2>Production stats (VM narrative cache)</h2>
__STATS__

<h2>Rule judge</h2>
__RULE_HITS__

<footer>
  Rendered nightly by the Narrative Eval workflow. CI cards refresh live from the
  public GitHub API; eval data comes from the latest nightly run.
</footer>
</main>
<script>
const DATA = __DATA_JSON__;
const RUNS_URL = 'https://api.github.com/repos/' + DATA.repo + '/actions/workflows/';

async function loadRun(file, cardId, label) {
  const card = document.getElementById(cardId);
  if (!card) return;
  const meta = card.querySelector('.wf-meta');
  const dot = card.querySelector('.dot');
  try {
    const res = await fetch(RUNS_URL + file + '/runs?per_page=1', { headers: { Accept: 'application/vnd.github+json' } });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    const run = (data.workflow_runs || [])[0];
    if (!run) throw new Error('no runs');
    const status = run.status === 'completed' ? run.conclusion : run.status;
    dot.className = 'dot ' + (status === 'success' ? 'ok' : status === 'failure' ? 'bad' : 'warn');
    const when = run.created_at ? new Date(run.created_at).toLocaleString() : '';
    meta.innerHTML = '<a href="' + run.html_url + '" target="_blank" rel="noopener">#' + run.run_number +
      '</a> · ' + (run.head_branch || '') + ' · ' + (run.head_sha || '').slice(0, 7) +
      '<br>' + status + (when ? ' · ' + when : '');
  } catch (err) {
    dot.className = 'dot mute';
    meta.textContent = 'unavailable (' + err.message + ')';
  }
}

DATA.workflows.forEach(function (wf) {
  loadRun(wf[0], 'wf-' + wf[0].replace('.', '-'), wf[1]);
});
</script>
</body>
</html>
"""
