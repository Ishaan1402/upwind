"""Human-reviewed labels for the LLM judge.

The deterministic validation set in ``backend.eval_validation`` is a good
sanity check, but the honest answer to "does the judge agree with a person?"
comes from labels a human actually writes. This module provides two commands:

    python -m backend.human_labels export --out labels.csv
        Dump recent cached narratives with their judge verdicts into a CSV
        with empty ``human_label`` and ``notes`` columns.

    python -m backend.human_labels validate --labels labels.csv --out validation.json
        Read the labeled CSV, compare human labels against the judge, and
        write the same JSON shape the eval dashboard expects.

Label values are ``pass``, ``fail``, or ``skip`` (case-insensitive). Skipped
rows do not count toward agreement or kappa.
"""

import argparse
import csv
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.db import DB_PATH, init_db
from backend.eval_validation import _cohens_kappa


def _load_cached_rows(db_path: str, limit: int) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT cache_key, narrative, judge_verdict_json, created_at "
        "FROM narrative_cache ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()

    out: List[Dict[str, Any]] = []
    for row in rows:
        verdict = None
        if row["judge_verdict_json"]:
            try:
                verdict = json.loads(row["judge_verdict_json"])
            except Exception:
                verdict = None
        out.append({
            "cache_key": row["cache_key"],
            "created_at": row["created_at"],
            "narrative": row["narrative"],
            "judge_verdict": (verdict or {}).get("verdict"),
            "judge_model": (verdict or {}).get("judge_model"),
        })
    return out


def export_rows(db_path: str, out_path: str, limit: int) -> int:
    """Write recent narratives to a labeling CSV. Returns the row count."""
    rows = _load_cached_rows(db_path, limit)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cache_key", "created_at", "narrative",
            "judge_verdict", "human_label", "notes",
        ])
        for row in rows:
            writer.writerow([
                row["cache_key"],
                row["created_at"],
                row["narrative"],
                row["judge_verdict"] or "",
                "",
                "",
            ])
    return len(rows)


def _binary_metrics(confusion: Dict[str, int]) -> Dict[str, Any]:
    """Per-class precision/recall/F1 and macro-F1 derived from confusion counts.

    Treats human labels as gold and judge verdicts as predicted, with ``pass``
    and ``fail`` as the two binary classes. Any metric whose denominator is
    zero (e.g. a class with no gold or no predicted instances) is None.
    """
    gp_jp = confusion["gold_pass_judge_pass"]  # gold pass, predicted pass
    gp_jf = confusion["gold_pass_judge_fail"]  # gold pass, predicted fail
    gf_jp = confusion["gold_fail_judge_pass"]  # gold fail, predicted pass
    gf_jf = confusion["gold_fail_judge_fail"]  # gold fail, predicted fail

    def _ratio(num: int, denom: int) -> Optional[float]:
        return round(num / denom, 4) if denom else None

    def _f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
        if precision is None or recall is None or (precision + recall) == 0:
            return None
        return round(2 * precision * recall / (precision + recall), 4)

    precision = {
        "pass": _ratio(gp_jp, gp_jp + gf_jp),
        "fail": _ratio(gf_jf, gf_jf + gp_jf),
    }
    recall = {
        "pass": _ratio(gp_jp, gp_jp + gp_jf),
        "fail": _ratio(gf_jf, gf_jf + gf_jp),
    }
    f1 = {
        "pass": _f1(precision["pass"], recall["pass"]),
        "fail": _f1(precision["fail"], recall["fail"]),
    }
    macro_f1 = (
        round((f1["pass"] + f1["fail"]) / 2, 4)
        if f1["pass"] is not None and f1["fail"] is not None
        else None
    )
    return {"precision": precision, "recall": recall, "f1": f1, "macro_f1": macro_f1}


def validate_labels(labels_path: str, db_path: str) -> Dict[str, Any]:
    """Compare human labels against cached judge verdicts."""
    judge_by_key = {r["cache_key"]: r for r in _load_cached_rows(db_path, 100000)}
    results: List[Dict[str, Any]] = []
    human_values: List[str] = []
    judge_values: List[str] = []

    with open(labels_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cache_key = (row.get("cache_key") or "").strip()
            human_raw = (row.get("human_label") or "").strip().lower()
            if not cache_key or human_raw not in ("pass", "fail", "skip"):
                continue
            judge_row = judge_by_key.get(cache_key, {})
            judge_value = judge_row.get("judge_verdict")
            agreement = None
            if human_raw in ("pass", "fail") and judge_value in ("pass", "fail"):
                agreement = human_raw == judge_value
                human_values.append(human_raw)
                judge_values.append(judge_value)
            results.append({
                "name": cache_key,
                "gold_verdict": human_raw,
                "judge_verdict": judge_value,
                "agreement": agreement,
                "judge_model": judge_row.get("judge_model"),
            })

    judged = sum(1 for r in results if r["agreement"] is not None)
    exact_agreement = (
        round(sum(1 for r in results if r["agreement"]) / judged, 4)
        if judged
        else None
    )
    confusion = {
        "gold_pass_judge_pass": sum(
            1 for g, p in zip(human_values, judge_values) if g == "pass" and p == "pass"
        ),
        "gold_pass_judge_fail": sum(
            1 for g, p in zip(human_values, judge_values) if g == "pass" and p == "fail"
        ),
        "gold_fail_judge_pass": sum(
            1 for g, p in zip(human_values, judge_values) if g == "fail" and p == "pass"
        ),
        "gold_fail_judge_fail": sum(
            1 for g, p in zip(human_values, judge_values) if g == "fail" and p == "fail"
        ),
    }

    metrics = _binary_metrics(confusion)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label_source": "human",
        "total_cases": len(results),
        "judged_cases": judged,
        "exact_agreement": exact_agreement,
        "cohens_kappa": _cohens_kappa(human_values, judge_values),
        "confusion": confusion,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "macro_f1": metrics["macro_f1"],
        "results": results,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="backend.human_labels", description=__doc__)
    parser.add_argument("--db", default=DB_PATH, help="SQLite path (default: backend/cache.db)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="dump recent narratives for labeling")
    export_parser.add_argument("--out", required=True, help="output CSV path")
    export_parser.add_argument("--limit", type=int, default=50, help="max rows to export")

    validate_parser = subparsers.add_parser("validate", help="compare labeled CSV against judge")
    validate_parser.add_argument("--labels", required=True, help="labeled CSV path")
    validate_parser.add_argument("--out", help="optional validation JSON output path")

    args = parser.parse_args(argv)
    init_db(args.db)

    if args.command == "export":
        count = export_rows(args.db, args.out, args.limit)
        print(f"exported {count} rows to {args.out}")
        return 0

    result = validate_labels(args.labels, args.db)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote validation results to {args.out}")
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
