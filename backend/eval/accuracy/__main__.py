"""CLI entrypoint for the offline accuracy-evaluation pipeline.

Run as: python -m backend.eval.accuracy ...

Subcommands:
  run     score + label every stored site-day in a date range (optionally
          scoped to a --bbox), persist one PredictionRecord per sample, and
          print the metrics summary plus a compact confusion matrix
  report  recompute the metrics summary over stored predictions
  label   compute ground-truth labels only (no scoring) and print the
          per-class distribution — a quick sanity check before a full run
  ingest  ingest historical raw data for a source (aqs/weather/hms/firms/all).
          Idempotent (INSERT OR REPLACE + skip-already-downloaded), resumable
          (per-source watermarks + per-step failure collection), and bbox-
          scoped; downloads land under ``RAW_DATA_DIR/<source>/<year>/``
  status  print per-source ingest watermarks, table row counts, and the
          min/max date in aqs_daily
"""

import argparse
import json
import sys
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from backend.eval.accuracy import RAW_DATA_DIR
from backend.eval.accuracy.ingest.aqs import ingest_aqs_year
from backend.eval.accuracy.ingest.firms_historical import ingest_firms_historical
from backend.eval.accuracy.ingest.hms import ingest_hms_smoke_range
from backend.eval.accuracy.ingest.weather import ingest_weather_for_sites
from backend.eval.accuracy.labels import LABEL_CLASSES
from backend.eval.accuracy.metrics import compute_metrics
from backend.eval.accuracy.runner import (
    PROGRESS_INTERVAL,
    build_samples,
    filter_sites_by_bbox,
    label_sample,
    run_accuracy_eval,
)
from backend.eval.accuracy.store import ACCURACY_DB_PATH, AccuracyStore

# Compact confusion-matrix column labels (7 columns stay printable).
_CONFUSION_ABBREV = {
    "wildfire_smoke": "smoke",
    "ozone_episode": "ozone",
    "windblown_dust": "dust",
    "winter_stagnation": "stagn",
    "urban_industrial_pm": "urban",
    "clean": "clean",
    "ambiguous": "ambig",
}

# Default FIRMS bbox: the contiguous US.
CONUS_BBOX = (-125.0, 24.0, -66.0, 50.0)

# Watermark source names (the `source` column of the ingest_state table).
_INGEST_SOURCES = ("aqs", "weather", "hms", "firms")

# Rolling-job lookback when a source has no watermark yet.
_INITIAL_HMS_LOOKBACK_DAYS = 30
_INITIAL_WEATHER_LOOKBACK_DAYS = 7
_FIRMS_NRT_WINDOW_DAYS = 7


def _date_type(value: str) -> str:
    """argparse type: accept YYYY-MM-DD dates only."""
    try:
        date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a valid date: {value!r}")
    return value


def _bbox_type(value: str) -> Tuple[float, float, float, float]:
    """argparse type: accept west,south,east,north float bboxes only."""
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"not a valid bbox (expected west,south,east,north): {value!r}"
        )
    try:
        west, south, east, north = (float(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"not a valid bbox (expected 4 floats): {value!r}"
        )
    return (west, south, east, north)


def _add_db_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db",
        default=ACCURACY_DB_PATH,
        help=f"SQLite store path (default: {ACCURACY_DB_PATH})",
    )


def _site_ids_for_bbox(
    store: AccuracyStore, bbox: Optional[Tuple[float, float, float, float]]
) -> Optional[List[str]]:
    """``site_ids`` whitelist covering the sites inside ``bbox``, or None when
    no bbox is given (unscoped = every site)."""
    if bbox is None:
        return None
    return [site_id for site_id, _, _ in filter_sites_by_bbox(store.fetch_aqs_sites(), bbox)]


def run_eval_command(
    db_path: str,
    start: str,
    end: str,
    limit: Optional[int] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, Any]:
    """Open the store and run the full eval, returning ``run_accuracy_eval``'s
    result dict. ``bbox`` restricts the samples to that region's site-days."""
    with AccuracyStore(db_path) as store:
        site_ids = _site_ids_for_bbox(store, bbox)
        return run_accuracy_eval(store, start, end, limit=limit, site_ids=site_ids)


def report_command(db_path: str) -> Dict[str, Any]:
    """Recompute the metrics summary over every stored prediction."""
    with AccuracyStore(db_path) as store:
        predictions = store.fetch_predictions()
        metrics = compute_metrics(
            (p.true_label, p.predicted_label) for p in predictions
        )
        return {"metrics": metrics, "samples": len(predictions)}


def label_command(
    db_path: str,
    start: str,
    end: str,
    limit: Optional[int] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, Any]:
    """Compute ground-truth labels only (no scoring) over the date range and
    return the per-class distribution. Used as a quick sanity check before a
    full evaluation run. ``bbox`` restricts the samples to that region's
    site-days."""
    with AccuracyStore(db_path) as store:
        site_ids = _site_ids_for_bbox(store, bbox)
        samples = build_samples(store, start, end, site_ids=site_ids)
        if limit is not None:
            samples = samples[:limit]
        counts: Dict[str, int] = {}
        total = len(samples)
        for i, (site_id, date_local) in enumerate(samples, start=1):
            label = label_sample(store, site_id, date_local).label
            counts[label] = counts.get(label, 0) + 1
            if i % PROGRESS_INTERVAL == 0:
                print(f"  processed {i}/{total}", file=sys.stderr)
        return {"samples": len(samples), "label_counts": counts}


# ---------------------------------------------------------------------------
# ingest subcommand
# ---------------------------------------------------------------------------


def _year_chunks(start: str, end: str) -> List[Tuple[int, Tuple[str, str]]]:
    """Split ``[start, end]`` (both ``YYYY-MM-DD``) into one ``(year, (seg_start,
    seg_end))`` chunk per calendar year, covering the window exactly once."""
    chunks: List[Tuple[int, Tuple[str, str]]] = []
    day = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    while day <= end_date:
        year_end = min(end_date, date(day.year, 12, 31))
        chunks.append((day.year, (day.isoformat(), year_end.isoformat())))
        day = date(year_end.year + 1, 1, 1)
    return chunks


def _hms_incremental_window(db_path: str) -> Tuple[str, str]:
    """``[start, end]`` for a rolling HMS run: from the source watermark (or
    30 days ago when none) through today."""
    end = date.today().isoformat()
    with AccuracyStore(db_path) as store:
        watermark = store.get_watermark("hms")
    start = watermark or (
        date.today() - timedelta(days=_INITIAL_HMS_LOOKBACK_DAYS)
    ).isoformat()
    return (min(start, end), end)


def _weather_incremental_window(db_path: str) -> Tuple[str, str]:
    """``[start, end]`` for a rolling weather run: from the source watermark
    (or a recent lookback when none) through today."""
    end = date.today().isoformat()
    with AccuracyStore(db_path) as store:
        watermark = store.get_watermark("weather")
    start = watermark or (
        date.today() - timedelta(days=_INITIAL_WEATHER_LOOKBACK_DAYS)
    ).isoformat()
    return (min(start, end), end)


def _ingest_aqs(args) -> int:
    """Ingest AQS daily summaries per year (idempotent; already-downloaded
    zips are skipped). Year-granular, so ``--incremental`` is a no-op."""
    if args.incremental:
        print(
            "aqs --incremental is a no-op: AQS is year-granular. Pass "
            "--year/--years to ingest; re-running a year is idempotent."
        )
        return 0
    years = sorted(set(args.year + (args.years or [])))
    if not years:
        print("error: 'ingest aqs' requires --year YYYY (repeatable) or --years YYYY YYYY")
        return 2
    failures: List[str] = []
    with AccuracyStore(args.db) as store:
        for year in years:
            try:
                out_dir = RAW_DATA_DIR / "aqs" / str(year)
                count = ingest_aqs_year(year, store, out_dir)
                store.set_watermark("aqs", f"{year}-12-31")
                print(f"aqs {year}: ingested {count} record(s) -> {out_dir}")
            except Exception as exc:  # keep going: a bad year must not block the rest
                print(f"aqs {year}: FAILED: {exc}", file=sys.stderr)
                failures.append(f"aqs:{year}: {exc}")
    if failures:
        print(f"failed: {failures}")
        return 1
    return 0


def _ingest_weather(args) -> int:
    """Ingest Open-Meteo daily weather for every AQS site (optionally bbox-
    scoped). The site list is derived from the AQS store, so AQS must be
    ingested first."""
    if args.incremental:
        start, end = _weather_incremental_window(args.db)
    else:
        start, end = args.start, args.end
    if start is None or end is None:
        print("error: 'ingest weather' requires --start and --end (or --incremental)")
        return 2
    with AccuracyStore(args.db) as store:
        sites = store.fetch_aqs_sites()
        if not sites:
            print(
                "error: no AQS data in the store — ingest AQS first "
                "(e.g. 'python -m backend.eval.accuracy ingest aqs --year 2020')"
            )
            return 1
        sites = filter_sites_by_bbox(sites, args.bbox)
        if not sites:
            print(f"error: no AQS sites inside bbox {args.bbox}")
            return 1
        try:
            total = ingest_weather_for_sites(
                sites,
                start,
                end,
                store,
                skip_existing=True,
                workers=args.workers,
            )
        except Exception as exc:
            print(f"weather [{start}..{end}]: FAILED: {exc}", file=sys.stderr)
            print(f"failed: ['weather: {exc}']")
            return 1
        store.set_watermark("weather", end)
    print(
        f"weather [{start}..{end}]: ingested {total} record(s) "
        f"across {len(sites)} site(s)"
    )
    return 0


def _ingest_hms(args) -> int:
    """Ingest NOAA HMS smoke polygons per day over the range, chunked by year
    so downloads land under ``RAW_DATA_DIR/hms/<year>/``. The adapter is
    already 404-tolerant per day; the year chunk is the CLI's failure step."""
    if args.incremental:
        start, end = _hms_incremental_window(args.db)
    else:
        start, end = args.start, args.end
    if start is None or end is None:
        print("error: 'ingest hms' requires --start and --end (or --incremental)")
        return 2
    failures: List[str] = []
    with AccuracyStore(args.db) as store:
        for year, (year_start, year_end) in _year_chunks(start, end):
            try:
                out_dir = RAW_DATA_DIR / "hms" / str(year)
                count = ingest_hms_smoke_range(year_start, year_end, store, out_dir)
                print(f"hms {year}: ingested {count} record(s) -> {out_dir}")
            except Exception as exc:  # keep going: a failed year must not block later years
                print(f"hms {year}: FAILED: {exc}", file=sys.stderr)
                failures.append(f"hms:{year}: {exc}")
        # Only advance the watermark when the whole range succeeded, so a
        # failed tail is retried by the next (incremental) run.
        if not failures:
            store.set_watermark("hms", end)
    if failures:
        print(f"failed: {failures}")
        return 1
    return 0


def _ingest_firms(args) -> int:
    """Ingest NASA FIRMS historical hotspots over the range, bbox-scoped
    (defaults to CONUS). In incremental mode the FIRMS area API's ~7-day NRT
    window is fetched, ignoring --start/--end."""
    if args.incremental:
        end = date.today().isoformat()
        start = (date.today() - timedelta(days=_FIRMS_NRT_WINDOW_DAYS - 1)).isoformat()
    else:
        start, end = args.start, args.end
    if start is None or end is None:
        print("error: 'ingest firms' requires --start and --end (or --incremental)")
        return 2
    bbox = args.bbox if args.bbox is not None else CONUS_BBOX
    with AccuracyStore(args.db) as store:
        try:
            count = ingest_firms_historical(start, end, bbox, store)
        except Exception as exc:
            print(f"firms [{start}..{end}]: FAILED: {exc}", file=sys.stderr)
            print(f"failed: ['firms: {exc}']")
            return 1
        store.set_watermark("firms", end)
    print(f"firms [{start}..{end}]: ingested {count} record(s)")
    return 0


def _ingest_all(args) -> int:
    """Run weather + hms + firms for the range (plus aqs when --year given).
    Each source is an independent step; failures are collected and reported
    after every source has finished."""
    if not args.incremental and (args.start is None or args.end is None):
        print("error: 'ingest all' requires --start and --end (or --incremental)")
        return 2
    codes: List[int] = []
    if args.incremental:
        codes.append(_ingest_aqs(args))  # prints the year-granular no-op message
    elif args.year or args.years:
        codes.append(_ingest_aqs(args))
    else:
        print("note: no --year given, skipping aqs (ingest aqs is year-granular)")
    codes.append(_ingest_weather(args))
    codes.append(_ingest_hms(args))
    codes.append(_ingest_firms(args))
    return 1 if any(code != 0 for code in codes) else 0


def ingest_command(args) -> int:
    """Orchestrate ``ingest <source>``; returns the process exit code."""
    handlers = {
        "aqs": _ingest_aqs,
        "weather": _ingest_weather,
        "hms": _ingest_hms,
        "firms": _ingest_firms,
        "all": _ingest_all,
    }
    return handlers[args.source](args)


# ---------------------------------------------------------------------------
# status subcommand
# ---------------------------------------------------------------------------


def status_command(db_path: str) -> Dict[str, Any]:
    """Per-source ingest watermarks, table row counts, and the min/max date in
    aqs_daily, as a plain dict (printing is left to the CLI)."""
    with AccuracyStore(db_path) as store:
        watermarks = {
            source: store.get_watermark(source) for source in _INGEST_SOURCES
        }
        row_counts = {
            "aqs_daily": store.count_aqs_daily(),
            "weather_daily": store.count_weather_daily(),
            "hms_smoke": store.count_hms_smoke(),
            "firms_hotspots": store.count_firms_hotspots(),
            "labels": store.count_labels(),
            "predictions": store.count_predictions(),
        }
        aqs_date_bounds = store.fetch_aqs_date_bounds()
    return {"watermarks": watermarks, "row_counts": row_counts, "aqs_date_bounds": aqs_date_bounds}


def _print_status(result: Dict[str, Any]) -> None:
    print("ingest watermarks (source -> last ingested date):")
    for source, last_date in result["watermarks"].items():
        print(f"  {source:<8} {last_date or '-'}")
    print("row counts:")
    for table, count in result["row_counts"].items():
        print(f"  {table:<16} {count}")
    bounds = result["aqs_date_bounds"]
    if bounds:
        print(f"aqs_daily date bounds: {bounds[0]} .. {bounds[1]}")
    else:
        print("aqs_daily date bounds: (no aqs_daily rows)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _print_confusion_matrix(confusion: Dict[str, Dict[str, int]]) -> None:
    labels = list(LABEL_CLASSES)
    headers = [_CONFUSION_ABBREV[label] for label in labels]
    width = max(len(h) for h in headers) + 2
    print("\nconfusion matrix (true rows x predicted columns):")
    print(f"{'true\\pred':<12}" + "".join(f"{h:>{width}}" for h in headers))
    for true_label in labels:
        row = confusion.get(true_label, {})
        counts = [row.get(predicted_label, 0) for predicted_label in labels]
        print(f"{true_label:<12}" + "".join(f"{c:>{width}}" for c in counts))


def _print_metrics(metrics: Dict[str, Any]) -> None:
    print(json.dumps(metrics, indent=2, default=str))
    _print_confusion_matrix(metrics["confusion"])
    notes = metrics.get("notes")
    if notes:
        print("\nnotes:")
        for note in notes:
            print(f"  - {note}")


def _add_ingest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "source",
        choices=["aqs", "weather", "hms", "firms", "all"],
        help="data source to ingest",
    )
    parser.add_argument(
        "--year", type=int, action="append", default=[],
        help="AQS year to ingest (repeatable: --year 2019 --year 2020)",
    )
    parser.add_argument(
        "--years", type=int, nargs="+", default=[],
        help="AQS years to ingest (e.g. --years 2019 2020)",
    )
    parser.add_argument(
        "--start", type=_date_type, default=None,
        help="inclusive start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", type=_date_type, default=None,
        help="inclusive end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--bbox", type=_bbox_type, default=None,
        help="bounding box west,south,east,north (weather/firms/all; "
             "firms defaults to CONUS)",
    )
    parser.add_argument(
        "--incremental", action="store_true",
        help="rolling mode: continue from the source watermark (firms: last "
             "7-day NRT window; aqs: no-op)",
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="accepted; v1 just re-attempts the same range (ingest is idempotent)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="number of concurrent weather fetch threads (default: 4)",
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="backend.eval.accuracy",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="score + label a date range and print metrics"
    )
    run_parser.add_argument(
        "--start", required=True, type=_date_type,
        help="inclusive start date (YYYY-MM-DD)",
    )
    run_parser.add_argument(
        "--end", required=True, type=_date_type,
        help="inclusive end date (YYYY-MM-DD)",
    )
    run_parser.add_argument(
        "--limit", type=int, default=None,
        help="cap the number of samples evaluated",
    )
    run_parser.add_argument(
        "--bbox", type=_bbox_type, default=None,
        help="evaluate only sites inside west,south,east,north",
    )
    _add_db_arg(run_parser)

    report_parser = subparsers.add_parser(
        "report", help="summarize stored predictions"
    )
    _add_db_arg(report_parser)

    label_parser = subparsers.add_parser(
        "label", help="compute labels only over a date range (class distribution)"
    )
    label_parser.add_argument(
        "--start", required=True, type=_date_type,
        help="inclusive start date (YYYY-MM-DD)",
    )
    label_parser.add_argument(
        "--end", required=True, type=_date_type,
        help="inclusive end date (YYYY-MM-DD)",
    )
    label_parser.add_argument(
        "--limit", type=int, default=None,
        help="cap the number of samples labeled",
    )
    label_parser.add_argument(
        "--bbox", type=_bbox_type, default=None,
        help="label only sites inside west,south,east,north",
    )
    _add_db_arg(label_parser)

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="ingest historical raw data (idempotent, resumable, bbox-scoped)",
    )
    _add_ingest_args(ingest_parser)
    _add_db_arg(ingest_parser)

    status_parser = subparsers.add_parser(
        "status", help="print ingest watermarks + row counts"
    )
    _add_db_arg(status_parser)

    args = parser.parse_args(argv)

    if args.command == "run":
        result = run_eval_command(
            args.db, args.start, args.end, limit=args.limit, bbox=args.bbox
        )
        _print_metrics(result["metrics"])
        print(f"\nevaluated {result['samples']} sample(s)")
    elif args.command == "report":
        result = report_command(args.db)
        _print_metrics(result["metrics"])
        print(f"\n{result['samples']} stored prediction(s)")
    elif args.command == "label":
        result = label_command(args.db, args.start, args.end, limit=args.limit, bbox=args.bbox)
        print(json.dumps(result, indent=2, default=str))
    elif args.command == "ingest":
        return ingest_command(args)
    elif args.command == "status":
        _print_status(status_command(args.db))
    return 0


if __name__ == "__main__":
    sys.exit(main())
