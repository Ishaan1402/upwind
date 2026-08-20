"""CLI entrypoint for the offline accuracy-evaluation pipeline.

Run as: python -m backend.eval.accuracy ...

Subcommands:
  run     score + label every stored site-day in a date range (optionally
          scoped to a --bbox), persist one PredictionRecord per sample, and
          print the metrics summary plus a compact confusion matrix
  report  recompute the metrics summary over stored predictions
  label   compute ground-truth labels only (no scoring) and print the
          per-class distribution — a quick sanity check before a full run
  ingest  ingest historical raw data for a source
          (aqs/weather/hms/firms/speciation/all). Idempotent (INSERT OR
          REPLACE + skip-already-downloaded), resumable (per-source
          watermarks + per-step failure collection), and bbox-scoped;
          downloads land under ``RAW_DATA_DIR/<source>/<year>/``
  status  print per-source ingest watermarks, table row counts, and the
          min/max date in aqs_daily
  speciate  cross-tab IMPROVE-speciation composition labels (answer key)
            against the stored rule-derived and scorer labels
  specbench  multi-year smoke-vs-dust ablation benchmark: does severity or a
            source feature drive smoke-vs-dust performance across years
            (leave-year-out, region-season holdout, severity strata,
            severity-matched subset, abstention curve)
  tune    Phase-2 coordinate search: sweep scorer thresholds over the site-
          level VAL holdout against frozen labels and print a ranked table

The ``speciate`` cross-tab validates the product's causal EVENT hypotheses
(``wildfire_smoke``, ``windblown_dust``) against COMPOSITION classes derived
from IMPROVE chemistry (``biomass_smoke``, ``mineral_dust``, ``mixed``). The
answer key says what the PM is MADE OF, not which event produced it; the
``SPECIATION_TO_PRODUCT`` mapping below documents how each composition class
relates to the product hypotheses. ``ozone_episode``, ``winter_stagnation``,
and ``urban_industrial_pm`` are NOT composition-validated by this key.
"""

import argparse
import json
import sys
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from backend.engine.params import LABEL_PARAMS
from backend.eval.accuracy import RAW_DATA_DIR
from backend.eval.accuracy.ingest.aqs import ingest_aqs_year
from backend.eval.accuracy.ingest.firms_historical import ingest_firms_historical
from backend.eval.accuracy.ingest.hms import ingest_hms_smoke_range
from backend.eval.accuracy.ingest.speciation import ingest_speciation_year
from backend.eval.accuracy.ingest.weather import ingest_weather_for_sites
from backend.eval.accuracy.labels import LABEL_CLASSES, build_observation
from backend.eval.accuracy.metrics import compute_metrics
from backend.eval.accuracy.reconstruct import score_sample
from backend.eval.accuracy.runner import (
    PROGRESS_INTERVAL,
    build_samples,
    filter_sites_by_bbox,
    label_sample,
    run_accuracy_eval,
)
from backend.eval.accuracy.speclabels import (
    SPECIATION_CLASSES,
    classify_speciation,
    derive_components,
)
from backend.eval.accuracy.store import ACCURACY_DB_PATH, AccuracyStore
from backend.eval.accuracy.tune import PHASE2_GRID, run_tune

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

# Composition class (speciation answer key) -> the product hypothesis
# (LABEL_CLASSES id) whose causal event that composition can validate.
# Chemistry only says what the PM is MADE OF, so this map is where the
# composition vocabulary meets the product's event vocabulary:
#   biomass_smoke validates wildfire_smoke (biomass-combustion contribution;
#     wildfire smoke but also wood heating / ag burning, so it can only ever
#     support, never prove, the event hypothesis).
#   mineral_dust validates windblown_dust (fine-soil elements dominate; could
#     be windblown dust, road dust, construction, or resuspended soil).
#   mixed validates NEITHER pure claim: both biomass and mineral signatures
#     are elevated, so it is reported separately, never forced into either.
#   secondary_aerosol and ambiguous validate no product hypothesis here.
# ozone_episode / winter_stagnation / urban_industrial_pm are NOT
# composition-validated by this key (no composition class maps to them).
SPECIATION_TO_PRODUCT = {
    "biomass_smoke": "wildfire_smoke",
    "mineral_dust": "windblown_dust",
    "mixed": None,
    "secondary_aerosol": None,
    "ambiguous": None,
}

# Default FIRMS bbox: the contiguous US.
CONUS_BBOX = (-125.0, 24.0, -66.0, 50.0)

# Watermark source names (the `source` column of the ingest_state table).
_INGEST_SOURCES = ("aqs", "weather", "hms", "firms", "speciation")

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


def speciate_command(db_path: str) -> Dict[str, Any]:
    """Cross-tab IMPROVE-speciation composition labels against the stored
    rule-derived and scorer labels.

    For every ``(site_id, date_local)`` present in BOTH ``predictions`` and
    ``speciation``: derive the composition components from IMPROVE rows only
    (``speclabels.derive_components``), classify them
    (``speclabels.classify_speciation``), and compare against the stored
    ``true_label`` (rule-derived) and ``predicted_label`` (scorer).

    The speciation classes are COMPOSITION classes (what the PM is made of),
    not causal events; see ``SPECIATION_TO_PRODUCT`` for which product
    hypothesis each class validates. ``mixed`` days (both biomass and mineral
    signatures elevated) validate neither the pure-smoke nor the pure-dust
    claim and are reported separately. ``ozone_episode``, ``winter_stagnation``,
    and ``urban_industrial_pm`` are NOT composition-validated by this key.

    Returns a plain dict (printing is left to the CLI):
        joinable:       total site-days in both tables.
        elevated:       joinable site-days whose rule true_label is neither
                        ``clean`` nor ``ambiguous``.
        spec_distribution: speciation class -> count.
        crosstab_rule:  speciation class -> {rule true_label -> count}.
        crosstab_scorer: speciation class -> {scorer predicted_label -> count}.
        smoke_dust_scorer: smoke-vs-dust axis of the scorer cross-tab.
        smoke_dust_rule:   smoke-vs-dust axis of the rule-label cross-tab.
    """
    with AccuracyStore(db_path) as store:
        predictions = store.fetch_predictions()
        pred_by_key = {(p.site_id, p.date_local): p for p in predictions}
        joinable = store.fetch_speciation_join_site_days()

        crosstab_rule: Dict[str, Dict[str, int]] = {
            cls: {label: 0 for label in LABEL_CLASSES} for cls in SPECIATION_CLASSES
        }
        crosstab_scorer: Dict[str, Dict[str, int]] = {
            cls: {label: 0 for label in LABEL_CLASSES} for cls in SPECIATION_CLASSES
        }
        spec_distribution: Dict[str, int] = {cls: 0 for cls in SPECIATION_CLASSES}
        elevated = 0

        for site_id, date_local in joinable:
            pred = pred_by_key.get((site_id, date_local))
            if pred is None:
                continue
            components = derive_components(
                store.fetch_speciation(site_id, date_local),
                store.fetch_speciation_methods(site_id, date_local),
            )
            is_elevated = pred.true_label not in ("clean", "ambiguous")
            if is_elevated:
                elevated += 1
            spec_label = classify_speciation(components, elevated=is_elevated)

            spec_distribution[spec_label] += 1
            crosstab_rule[spec_label][pred.true_label] += 1
            crosstab_scorer[spec_label][pred.predicted_label] += 1

    def _axis(crosstab: Dict[str, Dict[str, int]]) -> Dict[str, int]:
        return {
            "speciation_dust_called_smoke": crosstab["mineral_dust"]["wildfire_smoke"],
            "speciation_smoke_called_dust": crosstab["biomass_smoke"]["windblown_dust"],
            "mixed_called_smoke": crosstab["mixed"]["wildfire_smoke"],
            "mixed_called_dust": crosstab["mixed"]["windblown_dust"],
        }

    return {
        "joinable": len(joinable),
        "elevated": elevated,
        "spec_distribution": spec_distribution,
        "crosstab_rule": crosstab_rule,
        "crosstab_scorer": crosstab_scorer,
        "smoke_dust_rule": _axis(crosstab_rule),
        "smoke_dust_scorer": _axis(crosstab_scorer),
    }


def speciate_pm_command(db_path: str) -> Dict[str, Any]:
    """PM-scoped speciation cross-tab with ON-THE-FLY production scoring.

    Extends ``speciate`` to the IMPROVE sites that only recently gained PM
    input (non-FRM 88502 mass) and weather: it starts from every site-day
    present in BOTH ``aqs_daily`` and ``speciation``, keeps only PM-elevated
    days (PM2.5 AQI > 50 OR PM10 AQI > 50), derives the speciation composition
    label (``speclabels``), and scores the day fresh with the production scorer
    (``reconstruct.score_sample``, the same path ``run_accuracy_eval`` uses) so
    the scorer sees PM mass + weather + FIRMS/HMS. Stored predictions are not
    used (they may predate the 88502/weather backfill) and are not written.

    The speciation classes are COMPOSITION classes; see ``SPECIATION_TO_PRODUCT``
    for which product hypothesis each validates. ``mixed`` days validate neither
    pure claim and are reported separately. ``ozone_episode``,
    ``winter_stagnation``, and ``urban_industrial_pm`` are NOT
    composition-validated by this key.

    Returns a plain dict (printing is left to the CLI):
        pm_elevated:       PM-elevated site-day count (the sample size).
        spec_distribution: speciation class -> count over PM-elevated days.
        crosstab_scorer:   speciation class -> {scorer predicted_label -> count}.
        smoke_dust:        the smoke/dust headline: biomass_smoke days the
                           scorer calls wildfire_smoke, mineral_dust days it
                           calls windblown_dust, the two cross-errors, and the
                           mixed days (reported separately).
    """
    with AccuracyStore(db_path) as store:
        crosstab_scorer: Dict[str, Dict[str, int]] = {
            cls: {label: 0 for label in LABEL_CLASSES} for cls in SPECIATION_CLASSES
        }
        spec_distribution: Dict[str, int] = {cls: 0 for cls in SPECIATION_CLASSES}
        pm_elevated = 0

        for site_id, date_local in store.fetch_aqs_speciation_site_days():
            observation = build_observation(
                store.fetch_aqs_daily(site_id=site_id, date_local=date_local)
            )
            # PM elevation: either PM2.5 or PM10 daily AQI above the elevated
            # threshold (build_observation only records non-null AQIs).
            pm25_aqi = observation.pollutant_aqi.get("PM2.5", 0)
            pm10_aqi = observation.pollutant_aqi.get("PM10", 0)
            if pm25_aqi <= LABEL_PARAMS.aqi_elevated and pm10_aqi <= LABEL_PARAMS.aqi_elevated:
                continue
            pm_elevated += 1

            components = derive_components(
                store.fetch_speciation(site_id, date_local),
                store.fetch_speciation_methods(site_id, date_local),
            )
            spec_label = classify_speciation(components, elevated=True)
            spec_distribution[spec_label] += 1

            observation_dict, hypotheses, _ = score_sample(store, site_id, date_local)
            if hypotheses:
                predicted_label = hypotheses[0]["id"]
            else:
                # The scorer could not commit to any cause; record as ambiguous
                # (mirrors run_accuracy_eval).
                predicted_label = "ambiguous"
            # Clean mapping identical to run_accuracy_eval: a non-elevated-AQI
            # day is clean regardless of the top hypothesis (never fires for
            # the PM-elevated set, kept for definitional consistency).
            if (
                observation_dict.get("aqi") is not None
                and observation_dict["aqi"] <= LABEL_PARAMS.aqi_elevated
            ):
                predicted_label = "clean"
            crosstab_scorer[spec_label][predicted_label] += 1

    return {
        "pm_elevated": pm_elevated,
        "spec_distribution": spec_distribution,
        "crosstab_scorer": crosstab_scorer,
        "smoke_dust": {
            "smoke_smoke": crosstab_scorer["biomass_smoke"]["wildfire_smoke"],
            "dust_dust": crosstab_scorer["mineral_dust"]["windblown_dust"],
            "dust_smoke": crosstab_scorer["mineral_dust"]["wildfire_smoke"],
            "smoke_dust": crosstab_scorer["biomass_smoke"]["windblown_dust"],
            "mixed": {
                "days": sum(crosstab_scorer["mixed"].values()),
                "called_smoke": crosstab_scorer["mixed"]["wildfire_smoke"],
                "called_dust": crosstab_scorer["mixed"]["windblown_dust"],
            },
        },
    }


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
# tune subcommand
# ---------------------------------------------------------------------------

# Named tuning grids: a grid name maps to a list of (name, overrides) combos
# for ``run_tune``. Only known grids are accepted so a typo never silently
# tunes nothing.
_TUNE_GRIDS = {"phase2": PHASE2_GRID}

# Ranked-table per-class F1 columns: (column header, LABEL_CLASSES key).
_TUNE_F1_COLUMNS = (
    ("smoke", "wildfire_smoke"),
    ("dust", "windblown_dust"),
    ("urban", "urban_industrial_pm"),
    ("stagn", "winter_stagnation"),
    ("ozone", "ozone_episode"),
)


def _fmt_metric(value: Optional[float]) -> str:
    """None-safe 3-decimal metric formatter for the ranked tune table."""
    return "-" if value is None else f"{value:.3f}"


def tune_command(
    db_path: str,
    start: str,
    end: str,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    limit: Optional[int] = None,
    val_fraction: float = 0.2,
    grid: str = "phase2",
) -> List[Dict[str, Any]]:
    """Run the Phase-2 coordinate search: sweep scorer thresholds over the
    site-level VAL holdout of the in-scope sites against frozen labels, and
    return the ranked result rows (printing is left to the CLI). Unknown grid
    names raise ``ValueError``."""
    if grid not in _TUNE_GRIDS:
        raise ValueError(
            f"unknown grid {grid!r} (choose from {sorted(_TUNE_GRIDS)})"
        )
    with AccuracyStore(db_path) as store:
        return run_tune(
            store,
            start,
            end,
            bbox=bbox,
            limit=limit,
            val_fraction=val_fraction,
            grid=_TUNE_GRIDS[grid],
        )


def _print_tune_table(rows: List[Dict[str, Any]]) -> None:
    """Print the ranked tune results as a readable table.

    Columns: name | non_clean_top1 | macro_f1 | smoke/dust/urban/stagnation/
    ozone F1 | overrides. Missing metrics render as ``-``.
    """
    header = (
        f"{'name':<30}"
        f"{'non_clean_top1':>14}{'macro_f1':>10}"
        + "".join(f"{header:>10}" for header, _ in _TUNE_F1_COLUMNS)
        + "  overrides"
    )
    print(header)
    for row in rows:
        per_class = row["per_class"]
        line = (
            f"{row['name']:<30}"
            f"{_fmt_metric(row['non_clean_top1_accuracy']):>14}"
            f"{_fmt_metric(row['macro_f1']):>10}"
        )
        for _, cls in _TUNE_F1_COLUMNS:
            line += f"{_fmt_metric(per_class.get(cls, {}).get('f1')):>10}"
        overrides = row["params"]
        line += f"  {overrides if overrides else '-'}"
        print(line)


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


def _advance_watermark(store: AccuracyStore, source: str, new_end: str) -> str:
    """Advance ``source``'s watermark to ``max(existing, new_end)``.

    Watermarks must never move backwards: ingesting an OLDER historical range
    on top of newer stored data (e.g. backfilling AQS 2016-2019 after 2020-2021
    rows are already present) must not regress the source's last-ingested
    date, or the next ``--incremental`` run would re-ingest the newer data.
    ``date_local`` is always stored as ``YYYY-MM-DD`` text, so lexical
    ordering compares the dates correctly. Returns the resulting watermark.
    """
    existing = store.get_watermark(source)
    last_date = new_end if existing is None else max(existing, new_end)
    store.set_watermark(source, last_date)
    return last_date


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
                _advance_watermark(store, "aqs", f"{year}-12-31")
                print(f"aqs {year}: ingested {count} record(s) -> {out_dir}")
            except Exception as exc:  # keep going: a bad year must not block the rest
                print(f"aqs {year}: FAILED: {exc}", file=sys.stderr)
                failures.append(f"aqs:{year}: {exc}")
    if failures:
        print(f"failed: {failures}")
        return 1
    return 0


def _ingest_speciation(args) -> int:
    """Ingest EPA AirData PM2.5 speciation for a single --year (idempotent;
    already-downloaded zips are skipped). Year-granular, so ``--incremental``
    is a no-op."""
    if args.incremental:
        print(
            "speciation --incremental is a no-op: speciation is year-granular. "
            "Pass --year to ingest; re-running a year is idempotent."
        )
        return 0
    years = sorted(set(args.year + (args.years or [])))
    if not years:
        print("error: 'ingest speciation' requires --year YYYY")
        return 2
    if len(years) > 1:
        print(f"note: speciation ingests one year at a time; using {years[0]}")
    year = years[0]
    with AccuracyStore(args.db) as store:
        try:
            out_dir = RAW_DATA_DIR / "speciation" / str(year)
            count = ingest_speciation_year(year, store, out_dir)
            _advance_watermark(store, "speciation", f"{year}-12-31")
            print(f"speciation {year}: ingested {count} record(s) -> {out_dir}")
        except Exception as exc:
            print(f"speciation {year}: FAILED: {exc}", file=sys.stderr)
            return 1
    return 0


def _ingest_weather(args) -> int:
    """Ingest Open-Meteo daily weather for every AQS site (optionally bbox-
    scoped). The site list is derived from the AQS store, so AQS must be
    ingested first.

    With ``--speciation`` the site list comes from the ``speciation_sites``
    table instead (distinct IMPROVE/CSN speciation sites with their SPEC-CSV
    coordinates), so weather can be backfilled for sites that never appear in
    the AQS daily summaries (e.g. IMPROVE speciation sites reporting PM2.5
    mass under the non-FRM 88502 code). This is a fixed-range backfill, so
    ``--incremental`` is rejected and the ``weather`` watermark is left
    untouched (advancing it would make a later incremental AQS-site run skip
    the range).
    """
    if args.speciation:
        if args.incremental:
            print(
                "error: 'ingest weather --speciation' is a fixed-range backfill; "
                "pass --start and --end"
            )
            return 2
        start, end = args.start, args.end
        if start is None or end is None:
            print(
                "error: 'ingest weather --speciation' requires --start and --end"
            )
            return 2
        with AccuracyStore(args.db) as store:
            sites = store.fetch_speciation_sites()
            if not sites:
                print(
                    "error: no speciation sites in the store — ingest "
                    "speciation first "
                    "(e.g. 'python -m backend.eval.accuracy ingest speciation "
                    "--year 2020')"
                )
                return 1
            sites = filter_sites_by_bbox(sites, args.bbox)
            if not sites:
                print(f"error: no speciation sites inside bbox {args.bbox}")
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
        print(
            f"weather (speciation sites) [{start}..{end}]: ingested {total} "
            f"record(s) across {len(sites)} site(s)"
        )
        return 0

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
        _advance_watermark(store, "weather", end)
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
            _advance_watermark(store, "hms", end)
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
        _advance_watermark(store, "firms", end)
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
        "speciation": _ingest_speciation,
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
            "speciation": store.count_speciation(),
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


# Compact column labels for the speciation cross-tabs.
_SPECIATION_ABBREV = {
    "biomass_smoke": "smoke",
    "mineral_dust": "dust",
    "mixed": "mixed",
    "secondary_aerosol": "second",
    "ambiguous": "ambig",
}


def _print_speciate_crosstab(crosstab: Dict[str, Dict[str, int]]) -> None:
    """Print a speciation (rows) x label-class (cols) count table."""
    headers = [_CONFUSION_ABBREV[label] for label in LABEL_CLASSES]
    width = max(len(h) for h in headers) + 2
    print(f"{'spec\\label':<12}" + "".join(f"{h:>{width}}" for h in headers))
    for spec_class in SPECIATION_CLASSES:
        row = crosstab.get(spec_class, {})
        counts = [row.get(label, 0) for label in LABEL_CLASSES]
        print(
            f"{_SPECIATION_ABBREV[spec_class]:<12}"
            + "".join(f"{c:>{width}}" for c in counts)
        )


def _print_speciate(result: Dict[str, Any]) -> None:
    """Print the speciation cross-tab report (see ``speciate_command``)."""
    print(f"joinable site-days (predictions x speciation): {result['joinable']}")
    print(f"  elevated (rule true_label not clean/ambiguous): {result['elevated']}")
    print(f"  non-elevated: {result['joinable'] - result['elevated']}")

    print("\nspeciation COMPOSITION distribution (IMPROVE chemistry, answer key):")
    for spec_class in SPECIATION_CLASSES:
        print(f"  {spec_class:<18} {result['spec_distribution'].get(spec_class, 0)}")

    print("\ncross-tab: speciation composition (rows) x rule true_label (cols):")
    _print_speciate_crosstab(result["crosstab_rule"])
    print("\ncross-tab: speciation composition (rows) x scorer predicted_label (cols):")
    _print_speciate_crosstab(result["crosstab_scorer"])

    rule = result["smoke_dust_rule"]
    scorer = result["smoke_dust_scorer"]
    print("\nsmoke-vs-dust axis (speciation composition x rule label):")
    print(f"  mineral-dust days the RULE calls wildfire_smoke:  {rule['speciation_dust_called_smoke']}")
    print(f"  biomass-smoke days the RULE calls windblown_dust: {rule['speciation_smoke_called_dust']}")
    print(f"  mixed days the RULE calls wildfire_smoke/windblown_dust: "
          f"{rule['mixed_called_smoke']}/{rule['mixed_called_dust']}")
    print("smoke-vs-dust axis (speciation composition x scorer prediction):")
    print(f"  mineral-dust days the SCORER calls wildfire_smoke:  {scorer['speciation_dust_called_smoke']}")
    print(f"  biomass-smoke days the SCORER calls windblown_dust: {scorer['speciation_smoke_called_dust']}")
    print(f"  mixed days the SCORER calls wildfire_smoke/windblown_dust: "
          f"{scorer['mixed_called_smoke']}/{scorer['mixed_called_dust']}")
    print("note: ozone_episode / winter_stagnation / urban_industrial_pm are NOT")
    print("  composition-validated by this answer key (no composition class maps to them).")


def _print_speciate_pm(result: Dict[str, Any]) -> None:
    """Print the PM-scoped speciation report (see ``speciate_pm_command``)."""
    print(
        f"PM-elevated IMPROVE site-days (PM2.5 AQI > 50 OR PM10 AQI > 50): "
        f"{result['pm_elevated']}"
    )

    print("\nspeciation COMPOSITION distribution (PM-elevated days, answer key):")
    for spec_class in SPECIATION_CLASSES:
        print(f"  {spec_class:<18} {result['spec_distribution'].get(spec_class, 0)}")

    print("\ncross-tab: speciation composition (rows) x scorer predicted_label "
          "(cols, scored fresh from archive):")
    _print_speciate_crosstab(result["crosstab_scorer"])

    sd = result["smoke_dust"]
    print("\nsmoke-vs-dust axis (speciation composition x fresh scorer prediction):")
    print(f"  biomass-smoke days the SCORER calls wildfire_smoke: {sd['smoke_smoke']}")
    print(f"  mineral-dust days the SCORER calls windblown_dust:   {sd['dust_dust']}")
    print(f"  mineral-dust days the SCORER calls wildfire_smoke:   {sd['dust_smoke']}")
    print(f"  biomass-smoke days the SCORER calls windblown_dust:  {sd['smoke_dust']}")
    mixed = sd["mixed"]
    print(f"  mixed days (both signatures elevated; reported separately, "
          f"never forced into either): {mixed['days']}")
    print(f"    ... the SCORER calls wildfire_smoke: {mixed['called_smoke']}")
    print(f"    ... the SCORER calls windblown_dust: {mixed['called_dust']}")
    print("note: ozone_episode / winter_stagnation / urban_industrial_pm are NOT")
    print("  composition-validated by this answer key (no composition class maps to them).")


def _add_ingest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "source",
        choices=["aqs", "weather", "hms", "firms", "speciation", "all"],
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
        "--speciation", action="store_true",
        help="weather only: backfill archived weather for IMPROVE/CSN "
             "speciation sites (from the speciation_sites table) instead of "
             "AQS sites; requires --start/--end and does not touch the weather "
             "watermark",
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

    tune_parser = subparsers.add_parser(
        "tune",
        help="sweep scorer thresholds over the val holdout vs frozen labels",
    )
    tune_parser.add_argument(
        "--start", required=True, type=_date_type,
        help="inclusive start date (YYYY-MM-DD)",
    )
    tune_parser.add_argument(
        "--end", required=True, type=_date_type,
        help="inclusive end date (YYYY-MM-DD)",
    )
    tune_parser.add_argument(
        "--bbox", type=_bbox_type, default=None,
        help="tune only sites inside west,south,east,north",
    )
    tune_parser.add_argument(
        "--limit", type=int, default=None,
        help="cap the number of val samples scored per combo",
    )
    tune_parser.add_argument(
        "--val-fraction", type=float, default=0.2,
        help="fraction of sites held out for validation (default: 0.2)",
    )
    tune_parser.add_argument(
        "--grid", type=str, default="phase2",
        help=f"tuning grid name (default: phase2; choices: {sorted(_TUNE_GRIDS)})",
    )
    _add_db_arg(tune_parser)

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

    speciate_parser = subparsers.add_parser(
        "speciate",
        help="cross-tab IMPROVE-speciation labels vs stored rule/scorer labels "
             "(--scored: PM-scoped cross-tab scored fresh from the archive)",
    )
    speciate_parser.add_argument(
        "--scored", action="store_true",
        help="PM-scoped cross-tab: for PM-elevated IMPROVE site-days (PM2.5 or "
             "PM10 AQI > 50) derive the speciation answer key and score the day "
             "fresh with the production scorer instead of using stored "
             "predictions",
    )
    _add_db_arg(speciate_parser)

    specbench_parser = subparsers.add_parser(
        "specbench",
        help="multi-year smoke-vs-dust ablation benchmark (severity vs source "
             "signal): leave-year-out, region-season holdout, severity strata, "
             "severity-matched subset, abstention curve",
    )
    _add_db_arg(specbench_parser)

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
    elif args.command == "tune":
        try:
            rows = tune_command(
                args.db,
                args.start,
                args.end,
                bbox=args.bbox,
                limit=args.limit,
                val_fraction=args.val_fraction,
                grid=args.grid,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if rows:
            print(
                f"tuned {len(rows)} param combo(s) over "
                f"{rows[0]['total']} val site-day sample(s) "
                f"({rows[0]['elevated_count']} elevated)"
            )
            _print_tune_table(rows)
        else:
            print("tune: no val-holdout samples in range; nothing to rank")
    elif args.command == "ingest":
        return ingest_command(args)
    elif args.command == "status":
        _print_status(status_command(args.db))
    elif args.command == "speciate":
        if getattr(args, "scored", False):
            _print_speciate_pm(speciate_pm_command(args.db))
        else:
            _print_speciate(speciate_command(args.db))
    elif args.command == "specbench":
        # Imported here so the accuracy CLI stays importable even if numpy or
        # the benchmark's heavier imports are unavailable.
        from backend.eval.accuracy.specbench import specbench_command

        print(specbench_command(args.db))
    return 0


if __name__ == "__main__":
    sys.exit(main())
