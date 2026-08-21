"""
LLM narrative evaluation tooling (run as: python -m backend.eval ...)

Subcommands:
  stats          aggregate judge verdicts in SQLite (pass rate, failure
                 categories, per-hypothesis breakdown, judge models)
  rule-judge     deterministic jargon/header/tip checks over cached narratives
  export-fails   dump judge-failed narratives to JSON or CSV for review
  corpus         generate + judge narratives for the fixed scenario corpus
  judge-compare  compare the default judge vs JUDGE_MODEL candidate on the corpus
  workflow-status  fetch latest workflow run statuses for the evidence page
"""

import argparse
import asyncio
import csv
import json
import os
import re
import sqlite3
import sys
import urllib.request
from collections import Counter
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from backend.db import DB_PATH
from backend.eval_corpus import CORPUS
from backend.eval_report import REPO, WORKFLOWS, render_dashboard
from backend.engine.score import score_hypotheses
from backend.llm import generate_narrative_briefing
from backend.llm_judge import judge_narrative

BANNED_JARGON = [
    "AOD", "FIRMS", "HMS", "OpenAQ",
    "µg/m³", "μg/m³", "µg/m3", "ug/m3", "ppb", "ppm",
    "hypothesis score",
]
PERCENT_RE = re.compile(r"\b\d{1,3}\s?%")
HEADER_RES = [
    re.compile(r"(?m)^\s*#{1,6}\s"),
    re.compile(r"\*\*Briefing", re.IGNORECASE),
]
TIP_KEYWORDS = [
    "sensitive", "asthma", "mask", "windows closed",
    "outdoor", "activity", "lungs", "take it easy",
]


def rule_judge(narrative: str) -> Dict[str, Any]:
    """Cheap deterministic checks complementing the LLM judge."""
    low = narrative.lower()
    jargon = {term for term in BANNED_JARGON if term.lower() in low}
    jargon.update(m.group(0) for m in PERCENT_RE.finditer(narrative))
    return {
        "jargon": sorted(jargon),
        "has_disallowed_headers": any(r.search(narrative) for r in HEADER_RES),
        "has_tip_heuristic": any(k in low for k in TIP_KEYWORDS),
    }


def _load_rows(db_path: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT cache_key, narrative, payload_json, judge_verdict_json, created_at "
        "FROM narrative_cache"
    ).fetchall()
    conn.close()
    loaded = []
    for row in rows:
        item = dict(row)
        item["verdict"] = json.loads(item["judge_verdict_json"]) if item["judge_verdict_json"] else None
        item["payload"] = json.loads(item["payload_json"]) if item["payload_json"] else {}
        loaded.append(item)
    return loaded


def _top_hypothesis(payload: Dict[str, Any]) -> str:
    hypotheses = payload.get("hypotheses") or []
    return (hypotheses[0] or {}).get("id", "unknown") if hypotheses else "unknown"


def compute_stats(db_path: str) -> Dict[str, Any]:
    rows = [r for r in _load_rows(db_path) if r["verdict"]]
    counts = Counter(r["verdict"].get("verdict") for r in rows)
    judged = counts["pass"] + counts["fail"]
    hallucinations = Counter(
        h for r in rows for h in (r["verdict"].get("hallucinations") or [])
    )
    jargon = Counter(
        j for r in rows for j in (r["verdict"].get("leaked_jargon") or [])
    )
    return {
        "total": len(rows),
        "verdict_counts": dict(counts),
        "pass_rate": round(counts["pass"] / judged, 3) if judged else None,
        "top_hallucinations": hallucinations.most_common(5),
        "top_jargon": jargon.most_common(5),
        "by_top_hypothesis": dict(Counter(_top_hypothesis(r["payload"]) for r in rows)),
        "judge_models": dict(Counter(r["verdict"].get("judge_model") for r in rows)),
    }


def compute_rule_judge(db_path: str) -> List[Dict[str, Any]]:
    hits = []
    for row in _load_rows(db_path):
        result = rule_judge(row["narrative"])
        if result["jargon"] or result["has_disallowed_headers"] or not result["has_tip_heuristic"]:
            hits.append({
                "cache_key": row["cache_key"],
                "created_at": row["created_at"],
                **result,
            })
    return hits


def export_fails(db_path: str, out_path: str, fmt: str) -> List[Dict[str, Any]]:
    records = [
        {
            "cache_key": row["cache_key"],
            "created_at": row["created_at"],
            "narrative": row["narrative"],
            "verdict": row["verdict"],
        }
        for row in _load_rows(db_path)
        if row["verdict"] and row["verdict"].get("verdict") == "fail"
    ]
    if fmt == "csv":
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cache_key", "created_at", "narrative", "verdict_json"])
            for record in records:
                writer.writerow([
                    record["cache_key"],
                    record["created_at"],
                    record["narrative"],
                    json.dumps(record["verdict"]),
                ])
    else:
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
    return records


async def run_corpus() -> List[Dict[str, Any]]:
    results = []
    for scenario in CORPUS:
        observation = scenario["observation"]
        signals = scenario["signals"]
        hypotheses, open_questions = score_hypotheses(observation, signals)
        narrative = (await generate_narrative_briefing(
            scenario["location"], observation, signals, hypotheses, open_questions
        )).narrative
        evidence = {
            "location": scenario["location"],
            "observation": observation,
            "signals": signals,
            "hypotheses": hypotheses,
            "open_questions": open_questions,
            "narrative": narrative,
        }
        verdict = await judge_narrative(evidence, narrative)
        results.append({
            "scenario": scenario["name"],
            "verdict": verdict.to_dict(),
            "narrative": narrative,
            "top_hypothesis": _top_hypothesis(evidence),
        })
    return results


async def run_judge_compare(alt_model: str) -> List[Dict[str, Any]]:
    results = []
    for scenario in CORPUS:
        observation = scenario["observation"]
        signals = scenario["signals"]
        hypotheses, open_questions = score_hypotheses(observation, signals)
        narrative = (await generate_narrative_briefing(
            scenario["location"], observation, signals, hypotheses, open_questions
        )).narrative
        evidence = {
            "location": scenario["location"],
            "observation": observation,
            "signals": signals,
            "hypotheses": hypotheses,
            "open_questions": open_questions,
            "narrative": narrative,
        }
        default_verdict = await judge_narrative(evidence, narrative)
        with patch("backend.llm_judge.DEFAULT_JUDGE_MODEL", alt_model):
            alt_verdict = await judge_narrative(evidence, narrative)
        results.append({
            "scenario": scenario["name"],
            "default_verdict": default_verdict.to_dict(),
            "alt_verdict": alt_verdict.to_dict(),
        })
    return results


def _print_corpus(results: List[Dict[str, Any]]) -> None:
    print(f"{'scenario':<20} {'verdict':<8} {'model':<26} {'hall':<5} {'jargon':<6} reasoning")
    for result in results:
        verdict = result["verdict"]
        reasoning = (verdict.get("reasoning") or "").replace("\n", " ")[:60]
        print(
            f"{result['scenario']:<20} {str(verdict.get('verdict')):<8} "
            f"{str(verdict.get('judge_model')):<26} "
            f"{len(verdict.get('hallucinations') or []):<5} "
            f"{len(verdict.get('leaked_jargon') or []):<6} {reasoning}"
        )
    passed = sum(1 for r in results if r["verdict"].get("verdict") == "pass")
    print(f"\n{passed}/{len(results)} scenarios passed")


def _print_compare(results: List[Dict[str, Any]]) -> None:
    print(f"{'scenario':<20} {'default':<8} {'alt':<8} agree")
    agreement = 0
    for result in results:
        default = result["default_verdict"].get("verdict")
        alt = result["alt_verdict"].get("verdict")
        agree = default == alt
        agreement += int(agree)
        print(f"{result['scenario']:<20} {str(default):<8} {str(alt):<8} {'yes' if agree else 'no'}")
    print(f"\nagreement: {agreement}/{len(results)}")


def fetch_workflow_statuses(
    repo: str = REPO,
    workflows: Optional[List[tuple]] = None,
    token: Optional[str] = None,
    timeout: int = 8,
) -> List[Dict[str, Any]]:
    """Fetch the latest run per workflow from the GitHub API.

    Called server-side during the nightly render so the evidence page bakes
    statuses into the static HTML instead of depending on unauthenticated
    client-side API calls (rate-limited to 60/hr/IP). Never raises per
    workflow: failures record status "unavailable".
    """
    workflows = workflows or WORKFLOWS
    results: List[Dict[str, Any]] = []
    for file, label in workflows:
        entry: Dict[str, Any] = {"file": file, "label": label, "status": "unavailable"}
        try:
            url = f"https://api.github.com/repos/{repo}/actions/workflows/{file}/runs?per_page=1"
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/vnd.github+json",
                    "User-Agent": "upwind-eval",
                },
            )
            if token:
                req.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            run = (data.get("workflow_runs") or [{}])[0] or {}
            entry.update({
                "status": run.get("conclusion") or run.get("status") or "unknown",
                "run_number": run.get("run_number"),
                "branch": run.get("head_branch"),
                "sha": (run.get("head_sha") or "")[:7],
                "html_url": run.get("html_url"),
                "created_at": run.get("created_at"),
            })
        except Exception as e:
            entry["status"] = "unavailable"
            entry["error"] = str(e)[:120]
        results.append(entry)
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="backend.eval", description=__doc__)
    parser.add_argument("--db", default=DB_PATH, help="SQLite cache path (default: backend/cache.db)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stats = subparsers.add_parser("stats", help="aggregate judge verdicts")
    stats.add_argument("--out", help="optional JSON output path")

    rule = subparsers.add_parser("rule-judge", help="deterministic checks over cached narratives")
    rule.add_argument("--out", help="optional JSON output path")

    export = subparsers.add_parser("export-fails", help="dump judge-failed narratives")
    export.add_argument("--out", required=True, help="output file path")
    export.add_argument("--format", choices=["json", "csv"], default="json")

    corpus = subparsers.add_parser("corpus", help="evaluate the fixed scenario corpus")
    corpus.add_argument("--out", help="optional JSON output path")

    compare = subparsers.add_parser("judge-compare", help="compare judges on the corpus")
    compare.add_argument("--model", required=True, help="candidate judge model (e.g. JUDGE_MODEL value)")
    compare.add_argument("--out", help="optional JSON output path")

    wf = subparsers.add_parser("workflow-status", help="fetch latest workflow run statuses for the evidence page")
    wf.add_argument("--out", required=True, help="output JSON path")

    dashboard = subparsers.add_parser(
        "render-dashboard",
        help="build the static eval dashboard HTML (publishes to VM via nightly workflow)",
    )
    dashboard.add_argument("--corpus", help="JSON file from `corpus --out`")
    dashboard.add_argument("--compare", help="JSON file from `judge-compare --out`")
    dashboard.add_argument("--stats", help="JSON file from `stats --out`")
    dashboard.add_argument("--rule-hits", help="JSON file from `rule-judge --out`")
    dashboard.add_argument("--metrics", help="JSON file from `python -m backend.metrics report --out`")
    dashboard.add_argument("--validation", help="JSON file from `python -m backend.eval_validation --out`")
    dashboard.add_argument("--label-validation", help="JSON file from `python -m backend.human_labels validate --out` (agent/human labels on live narratives)")
    dashboard.add_argument("--workflows", help="JSON file from `workflow-status --out` (baked CI statuses)")
    dashboard.add_argument("--public", action="store_true", help="hide narratives/cache keys for a public page")
    dashboard.add_argument("--out", required=True, help="output HTML path")

    args = parser.parse_args(argv)

    if args.command == "stats":
        result = compute_stats(args.db)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"wrote stats to {args.out}")
        else:
            print(json.dumps(result, indent=2, default=str))
    elif args.command == "rule-judge":
        hits = compute_rule_judge(args.db)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(hits, f, indent=2, default=str)
            print(f"wrote {len(hits)} rule violations to {args.out}")
        for hit in hits:
            print(hit)
        print(f"\n{len(hits)} narratives with rule violations")
    elif args.command == "export-fails":
        records = export_fails(args.db, args.out, args.format)
        print(f"exported {len(records)} failed narratives to {args.out}")
    elif args.command == "corpus":
        results = asyncio.run(run_corpus())
        _print_corpus(results)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
    elif args.command == "judge-compare":
        results = asyncio.run(run_judge_compare(args.model))
        _print_compare(results)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
    elif args.command == "workflow-status":
        results = fetch_workflow_statuses(token=os.getenv("GITHUB_TOKEN"))
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {len(results)} workflow statuses to {args.out}")
    elif args.command == "render-dashboard":
        def _load(path):
            if not path:
                return None
            with open(path) as f:
                return json.load(f)

        page = render_dashboard(
            corpus=_load(args.corpus),
            compare=_load(args.compare),
            stats=_load(args.stats),
            rule_hits=_load(args.rule_hits),
            metrics=_load(args.metrics),
            validation=_load(args.validation),
            label_validation=_load(args.label_validation),
            workflows=_load(args.workflows),
            public=args.public,
        )
        with open(args.out, "w") as f:
            f.write(page)
        print(f"rendered dashboard ({len(page)} bytes) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
