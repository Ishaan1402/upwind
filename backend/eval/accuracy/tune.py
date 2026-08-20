"""Phase-2 coordinate-search tuning for the accuracy-evaluation scorer.

Sweeps scorer thresholds against FROZEN ground-truth labels to find better
params for the smoke<->dust, smoke<->urban, and upwind sector/bonus
confusions. Pure logic: all I/O happens through the ``AccuracyStore`` passed
in, and the CLI (``__main__.py``) owns printing.

Ground-truth labels are derived ONCE with ``LABEL_PARAMS`` (frozen) and never
recomputed per combo, so a param sweep can only change the scorer's output,
never the labels it is measured against. Each combo swaps the active scorer
params via ``use_params(replace(Params(), **overrides))`` for its own scoring
pass; labels stay frozen on ``LABEL_PARAMS`` throughout.

The within-2020 leave-out is SITE-based: ``_site_holdout_split`` buckets every
site into train or val by its (deterministic) md5 hash, so a site's days are
never split across the partition (no site-day leakage).
"""

import hashlib
from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Tuple

from backend.engine.params import LABEL_PARAMS, Params, use_params
from backend.eval.accuracy.metrics import compute_metrics
from backend.eval.accuracy.runner import (
    build_samples,
    filter_sites_by_bbox,
    label_sample,
)
from backend.eval.accuracy.reconstruct import score_sample
from backend.eval.accuracy.store import AccuracyStore


def _site_holdout_split(
    site_ids: Iterable[str],
    val_fraction: float = 0.2,
) -> Tuple[List[str], List[str]]:
    """Deterministic site-based holdout split into ``(train_ids, val_ids)``.

    Each site is bucketed by ``int(md5(site_id).hexdigest(), 16) % 100``; a
    site goes to VAL when its bucket is below ``round(val_fraction * 100)``,
    else TRAIN. A site is entirely in one partition — its site-days never leak
    across the split. Deterministic, so re-running over the same site list
    returns the identical partition.
    """
    threshold = round(val_fraction * 100)
    train_ids: List[str] = []
    val_ids: List[str] = []
    for site_id in site_ids:
        digest = hashlib.md5(site_id.encode()).hexdigest()
        bucket = int(digest, 16) % 100
        (val_ids if bucket < threshold else train_ids).append(site_id)
    return train_ids, val_ids


def _predicted_label(observation: Dict, hypotheses: List[Dict]) -> str:
    """Mirror ``runner.run_accuracy_eval``'s prediction mapping.

    The scorer has no "clean" hypothesis id, so a non-elevated-AQI day is
    mapped to "clean" regardless of what the scorer ranks first (a missing AQI
    is NOT clean). ``LABEL_PARAMS`` (frozen) keeps this prediction-side
    definition identical to the label's clean definition.
    """
    if observation.get("aqi") is not None and observation["aqi"] <= LABEL_PARAMS.aqi_elevated:
        return "clean"
    if hypotheses:
        return hypotheses[0]["id"]
    return "ambiguous"


# Phase-2 coordinate-search grid: (name, {param: override}) combos centered on
# the frozen ``Params()`` baseline (the ORIGINAL untuned defaults), not the
# tuned ``DEFAULT``. Overrides are applied via
# ``dataclasses.replace(Params(), **overrides)``, so each row differs from the
# frozen baseline only on its named fields.
PHASE2_GRID: List[Tuple[str, Dict[str, float]]] = [
    ("baseline", {}),
    ("sector45", {"upwind_sector_width_deg": 45.0}),
    ("sector60", {"upwind_sector_width_deg": 60.0}),
    ("sector120", {"upwind_sector_width_deg": 120.0}),
    ("wfigs_bonus2", {"wfigs_upwind_bonus": 2.0}),
    ("wfigs_bonus1", {"wfigs_upwind_bonus": 1.0}),
    ("firms_bonus2", {"firms_upwind_bonus": 2.0}),
    ("firms_bonus1", {"firms_upwind_bonus": 1.0}),
    ("dust_max_025", {"openaq_dust_ratio_max": 0.25}),
    ("dust_max_045", {"openaq_dust_ratio_max": 0.45}),
    ("smoke_min_06", {"openaq_smoke_ratio_min": 0.60}),
    ("smoke_min_08", {"openaq_smoke_ratio_min": 0.80}),
    ("aod_haze_015", {"aod_haze": 0.15}),
    ("aod_haze_030", {"aod_haze": 0.30}),
    ("aod_medium_030", {"aod_medium": 0.30}),
    ("aod_medium_050", {"aod_medium": 0.50}),
    ("fire_min_2", {"fire_signal_min_count": 2}),
    ("highwind_12", {"high_wind_speed_mph": 12.0}),
    ("highwind_18", {"high_wind_speed_mph": 18.0}),
    ("sector45_bonus2", {"upwind_sector_width_deg": 45.0, "wfigs_upwind_bonus": 2.0, "firms_upwind_bonus": 2.0}),
    ("sector60_dust045_smoke06", {"upwind_sector_width_deg": 60.0, "openaq_dust_ratio_max": 0.45, "openaq_smoke_ratio_min": 0.60}),
    ("dust045_smoke06_highwind18", {"openaq_dust_ratio_max": 0.45, "openaq_smoke_ratio_min": 0.60, "high_wind_speed_mph": 18.0}),
]


def run_tune(
    store: AccuracyStore,
    start_date: str,
    end_date: str,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    limit: Optional[int] = None,
    val_fraction: float = 0.2,
    grid: Optional[List[Tuple[str, Dict[str, float]]]] = None,
) -> List[Dict]:
    """Coordinate-search over scorer thresholds against frozen labels.

    Resolves the sites in scope (``bbox`` when given), takes the VAL side of
    the deterministic site-based holdout, and scores every val site-day in
    ``[start_date, end_date]`` (capped at ``limit``) under each grid combo.
    True labels are computed ONCE with ``label_sample`` (which uses the frozen
    ``LABEL_PARAMS``) so no combo can move the labels.

    ``store`` must already be open (the CLI context-manages it); this function
    never opens or closes the store.

    Returns one row per combo, sorted by ``non_clean_top1_accuracy`` descending
    then ``macro_f1`` descending (None sorts last):
    ``{"name", "params" (overrides dict), "macro_f1", "non_clean_top1_accuracy",
    "per_class", "elevated_count", "total"}``.
    """
    sites = filter_sites_by_bbox(store.fetch_aqs_sites(), bbox)
    _, val_ids = _site_holdout_split(
        (site_id for site_id, _, _ in sites), val_fraction
    )

    samples = build_samples(store, start_date, end_date, site_ids=val_ids)
    if limit is not None:
        samples = samples[:limit]

    # Frozen ground truth: derived once, never recomputed per combo.
    true = {sample: label_sample(store, sample[0], sample[1]).label for sample in samples}

    combos = PHASE2_GRID if grid is None else grid
    rows: List[Dict] = []
    for name, overrides in combos:
        params = replace(Params(), **overrides)
        pairs: List[Tuple[str, str]] = []
        with use_params(params):
            for site_id, date_local in samples:
                observation, hypotheses, _ = score_sample(store, site_id, date_local)
                pairs.append(
                    (true[(site_id, date_local)], _predicted_label(observation, hypotheses))
                )
        metrics = compute_metrics(pairs)
        rows.append({
            "name": name,
            "params": overrides,
            "macro_f1": metrics["macro_f1"],
            "non_clean_top1_accuracy": metrics["non_clean_top1_accuracy"],
            "per_class": metrics["per_class"],
            "elevated_count": metrics["elevated_count"],
            "total": metrics["total"],
        })

    # None sorts last under the reverse sort: the bool "value is not None"
    # flips first, so a present value always outranks a missing one.
    return sorted(
        rows,
        key=lambda row: (
            row["non_clean_top1_accuracy"] is not None,
            row["non_clean_top1_accuracy"] if row["non_clean_top1_accuracy"] is not None else 0.0,
            row["macro_f1"] is not None,
            row["macro_f1"] if row["macro_f1"] is not None else 0.0,
        ),
        reverse=True,
    )
