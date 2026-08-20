"""Evaluation runner: tie the offline accuracy pipeline together.

Pure label derivation lives in ``labels``, scorer-input reconstruction in
``reconstruct``, and metrics in ``metrics``; this module is the I/O + orchestration
layer:

- ``build_samples`` enumerates the site-days to evaluate (distinct
  ``(site_id, date_local)`` pairs with any ``aqs_daily`` rows in the range,
  optionally restricted to a ``site_ids`` whitelist),
- ``filter_sites_by_bbox`` keeps the sites whose ``(lat, lon)`` fall inside a
  bounding box — the shared primitive behind ``--bbox`` on ``ingest``,
  ``run``, and ``label``,
- ``label_sample`` derives a site-day's ground-truth label from stored evidence
  (AQS daily summaries, weather, HMS smoke, FIRMS upwind fires),
- ``run_accuracy_eval`` scores every sample with the production scorer,
  persists one ``PredictionRecord`` per sample, and rolls the stored outcomes
  up into ``compute_metrics``.

Reproducibility guarantee
-------------------------
Per-sample evaluation is deterministic and idempotent: a sample
``(site_id, date_local)`` derives entirely from its own stored evidence, so
re-running ``run_accuracy_eval`` (or ``label``/``run``) over the same range
persists identical labels and predictions without duplication. A run scoped by
``--bbox`` and/or a date range is a faithful subset of a larger run: its
per-sample labels and predictions are exactly the corresponding subset of the
full run's, because ``build_samples``' whitelist only removes site-days and
never changes how an included sample is derived.
"""

import sys
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from backend.engine.params import LABEL_PARAMS
from backend.eval.accuracy.labels import build_observation, classify_sample
from backend.eval.accuracy.metrics import compute_metrics
from backend.eval.accuracy.records import LabelRecord, PredictionRecord
from backend.eval.accuracy.reconstruct import (
    _hms_res_from_store,
    _hotspots_for_day,
    _site_coords,
    score_sample,
)
from backend.eval.accuracy.store import AccuracyStore
from backend.services.firms import firms_search_radius_miles

# Prediction rows are inserted (and committed) in batches of this many.
INSERT_BATCH_SIZE = 500

# A progress line is printed to stderr every this many samples, so long
# evaluation/label runs show a heartbeat without spamming the terminal.
PROGRESS_INTERVAL = 500


def filter_sites_by_bbox(
    sites: Iterable[Tuple[str, float, float]],
    bbox: Optional[Tuple[float, float, float, float]],
) -> List[Tuple[str, float, float]]:
    """Keep ``(site_id, lat, lon)`` sites whose coordinates fall inside
    ``bbox`` ``(west, south, east, north)``, inclusive.

    ``None`` bbox keeps every site, so an unscoped run stays the union of all
    sites and a ``--bbox`` run is a strict subset. Sites with a missing
    coordinate are never included. Pure helper shared by the ``ingest`` and
    ``run``/``label`` CLIs.
    """
    if bbox is None:
        return list(sites)
    west, south, east, north = bbox
    return [
        (site_id, lat, lon)
        for site_id, lat, lon in sites
        if lat is not None
        and lon is not None
        and west <= lon <= east
        and south <= lat <= north
    ]


def build_samples(
    store: AccuracyStore,
    start_date: str,
    end_date: str,
    site_ids: Optional[Iterable[str]] = None,
) -> List[Tuple[str, str]]:
    """Distinct ``(site_id, date_local)`` pairs that have at least one
    ``aqs_daily`` row inside the inclusive ``[start_date, end_date]`` window,
    ordered deterministically by ``(site_id, date_local)``.

    When ``site_ids`` is given, only site-days whose site is in the whitelist
    are returned (the ``--bbox`` scoping path); ``None`` keeps every site, so
    an unscoped run remains the union of all site-days.
    """
    samples = store.fetch_aqs_site_days(start_date, end_date)
    if site_ids is not None:
        allowed = frozenset(site_ids)
        samples = [sample for sample in samples if sample[0] in allowed]
    return samples


def label_sample(
    store: AccuracyStore,
    site_id: str,
    date_local: str,
    rural: Optional[bool] = None,
) -> LabelRecord:
    """Derive the ground-truth label for one stored site-day.

    This is the I/O half of labeling: it pulls the day's AQS rows and
    ``WeatherDailyRecord`` out of the store and computes the fire/smoke context
    exactly as ``reconstruct`` does, then defers the classification rule to
    ``labels.classify_sample``:

    - ``smoke_density`` is the HMS density (``"light"``/``"medium"``/``"heavy"``)
      when ``_hms_res_from_store`` reports the site inside an archived plume,
    - ``upwind_fire`` is True when any surviving FIRMS pixel from
      ``_hotspots_for_day`` is upwind (wind direction from the weather record,
      search radius from ``firms_search_radius_miles``).
    """
    aqs_records = store.fetch_aqs_daily(site_id=site_id, date_local=date_local)
    weather_records = [
        r
        for r in store.fetch_weather_daily(site_id=site_id)
        if r.date_local == date_local
    ]
    weather = weather_records[0] if weather_records else None
    observation = build_observation(aqs_records)

    lat, lon = _site_coords(aqs_records, weather)
    smoke_density: Optional[str] = None
    upwind_fire: Optional[bool] = None
    if lat is not None and lon is not None:
        wind_speed = weather.wind_max_mph if weather is not None else None
        wind_dir = weather.wind_dir_dominant_deg if weather is not None else None
        radius_mi = firms_search_radius_miles(wind_speed, params=LABEL_PARAMS)

        hms_res = _hms_res_from_store(store, date_local, lat, lon)
        if hms_res.get("status") == "present":
            smoke_density = hms_res.get("density")

        pixels = _hotspots_for_day(
            store, lat, lon, date_local, wind_dir, radius_mi, params=LABEL_PARAMS
        )
        upwind_fire = any(p.get("is_upwind") is True for p in pixels)

    return classify_sample(
        observation,
        weather,
        smoke_density=smoke_density,
        upwind_fire=upwind_fire,
        rural=rural,
        site_id=site_id,
        date_local=date_local,
    )


def run_accuracy_eval(
    store: AccuracyStore,
    start_date: str,
    end_date: str,
    limit: Optional[int] = None,
    site_ids: Optional[Iterable[str]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    """Run the full offline evaluation over ``[start_date, end_date]``.

    For every stored site-day in the range (capped at ``limit`` when given;
    restricted to ``site_ids`` when given):
    reconstruct + score the day with the production scorer, derive its
    ground-truth label, and persist a ``PredictionRecord``. Rows are inserted
    in ``INSERT_BATCH_SIZE`` chunks. When ``progress`` is supplied it is called
    as ``progress(i, total)`` after each sample.

    A sample whose scorer returns no hypotheses is recorded with
    ``predicted_label="ambiguous"`` (and a None top score) rather than crashed.

    Returns ``{"metrics": ..., "samples": n}`` where the metrics are computed
    over every prediction currently stored (``store.fetch_predictions()``) and
    ``n`` is the number of samples processed in this run.
    """
    samples = build_samples(store, start_date, end_date, site_ids=site_ids)
    if limit is not None:
        samples = samples[:limit]

    total = len(samples)
    batch: List[PredictionRecord] = []
    for i, (site_id, date_local) in enumerate(samples, start=1):
        observation, hypotheses, _ = score_sample(store, site_id, date_local)
        if hypotheses:
            top = hypotheses[0]
            predicted_label = top["id"]
            top_score = top["score"]
            top_confidence = top["confidence"]
        else:
            # The scorer could not commit to any cause; record it as ambiguous.
            predicted_label = "ambiguous"
            top_score = None
            top_confidence = None

        # Clean mapping: the scorer has no "clean" hypothesis id, so a
        # non-elevated-AQI day is mapped to "clean" regardless of what the
        # scorer ranks first (a missing AQI is NOT clean). top_score and
        # top_confidence still record the top hypothesis for transparency.
        # LABEL_PARAMS (frozen) keeps this prediction-side definition identical
        # to the label's clean definition.
        if observation.get("aqi") is not None and observation["aqi"] <= LABEL_PARAMS.aqi_elevated:
            predicted_label = "clean"

        true = label_sample(store, site_id, date_local)
        batch.append(
            PredictionRecord(
                site_id=site_id,
                date_local=date_local,
                true_label=true.label,
                predicted_label=predicted_label,
                top_score=top_score,
                top_confidence=top_confidence,
            )
        )
        if len(batch) >= INSERT_BATCH_SIZE:
            store.insert_predictions(batch)
            batch = []
        if i % PROGRESS_INTERVAL == 0:
            print(f"  processed {i}/{total}", file=sys.stderr)
        if progress is not None:
            progress(i, total)

    if batch:
        store.insert_predictions(batch)

    predictions = store.fetch_predictions()
    metrics = compute_metrics(
        (p.true_label, p.predicted_label) for p in predictions
    )
    return {"metrics": metrics, "samples": total}
