"""Multi-year smoke-vs-dust ablation benchmark (severity-vs-source test).

This is the evaluation that asks whether ANY current feature carries stable
source signal across years, or whether the earlier (2020-only) finding holds:
that a smoke-vs-dust classifier's performance was driven almost entirely by PM
severity (``pm_aqi``), not by source signals.

What it does
------------
1. Assembles a binary dataset from the accuracy store: every ``(site_id,
   date_local)`` joinable across ``speciation`` x ``aqs_daily`` that is
   PM-elevated (max AQI over 88101 / 88502 / 81102 > 50) and whose IMPROVE
   chemistry confidently classifies as ``biomass_smoke`` (=1) or
   ``mineral_dust`` (=0) using the ``speclabels`` answer key (IMPROVE method
   filter only; ``mixed`` / ``secondary_aerosol`` / ``ambiguous`` are counted
   and excluded from the binary task). No smoke/dust rule is re-derived here.
2. Fits L2-regularized logistic regressions over NON-CHEMICAL inference
   features: ``pm_aqi`` (severity), ``wind_max_mph`` (imputed with the
   training-fold median when missing), ``pm25_pm10_ratio`` + a
   ``ratio_missing`` indicator (via ``reconstruct._openaq_sig_from_aqs``),
   month as sin/cos season, a 6-region grouping of the Western states,
   — Track B — daily max gust (``gust_max_mph``) + antecedent 30-day
   precipitation (``precip_30d_in``, the blowing-dust / Lamar climatology),
   and — new — 850 hPa transport-layer wind as ``t850_speed`` plus circular
   ``t850_dir_sin``/``t850_dir_cos`` (NCEP/NCAR Reanalysis-1 daily 850 hPa
   u/v at 2.5°, stored in ``transport_wind``; ``t850_missing`` flags gaps).
3. Evaluates 11 ablations (severity_only / wind_only / ratio_only /
   season_region_only / t850_only / dust_opportunity / all / all_plus_t850 /
   all_plus_dust / all_minus_severity / all_minus_season_region)
   under four regimes:
     - leave-year-out (train 2016..2020 except Y, evaluate on Y),
     - a region-season holdout (the dust-heaviest region-season in the data),
     - severity strata (pm_aqi terciles cut on the training fold),
     - a severity-matched subset (greedy nearest-severity dust per smoke),
   plus an abstention curve for the ``all`` model and a no-model hard-rule
   diagnostic (the literal Lamar rule: gust >= 40 mph AND antecedent 30-day
   precip <= 0.6 in -> dust).

Honesty mechanics
-----------------
- Standardization and median imputation are fit on the TRAINING fold only
  (never on the test fold), and severity tercile cut points come from the
  training fold too.
- Balanced accuracy = mean of per-class recall; folds without both classes in
  the test set report None and are skipped in aggregates.
- Fire/HMS features are deliberately NOT included: multi-year FIRMS/HMS
  attribution is not ingested yet, so including it would either leak or be
  unavailable across the years this benchmark spans.
- This module is a benchmark only: it never writes predictions, never calls
  the production scorer (``score_sample``), and never touches ``Params`` or
  the ``speclabels`` thresholds.
"""

import math
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from backend.eval.accuracy.reconstruct import _openaq_sig_from_aqs
from backend.eval.accuracy.records import WeatherDailyRecord
from backend.eval.accuracy.speclabels import (
    SPECIATION_CLASSES,
    classify_speciation,
    derive_components,
)
from backend.eval.accuracy.store import AccuracyStore

# ---------------------------------------------------------------------------
# Dataset constants (documented; none of these are scorer/Params values)
# ---------------------------------------------------------------------------

# PM parameter codes whose daily AQI define PM elevation. 88101 is FRM/FEM
# PM2.5, 88502 the non-FRM PM2.5 mass code IMPROVE sites report under, and
# 81102 PM10 — exactly the codes the task scopes elevation to.
PM_PARAM_CODES: Tuple[str, ...] = ("88101", "88502", "81102")

# PM-elevated threshold (max AQI over the PM codes must be STRICTLY above 50,
# mirroring the task's "max AQI ... > 50").
PM_AQI_ELEVATED = 50

# The binary task: biomass smoke = 1, mineral dust = 0. Only these two
# composition classes enter the model; the others are counted separately.
POSITIVE_CLASS = "biomass_smoke"
NEGATIVE_CLASS = "mineral_dust"

# Default years for the leave-year-out splits (speciation coverage).
BENCHMARK_YEARS: Tuple[int, ...] = (2016, 2017, 2018, 2019, 2020)

# Greedy severity-match tolerance (AQI points): a smoke sample is matched to
# the nearest unmatched dust sample with |pm_aqi_dust - pm_aqi_smoke| <= tol.
SEVERITY_MATCH_TOL = 10.0

# Abstention margins (tau) for the abstention curve.
ABSTENTION_TAUS = (0.0, 0.1, 0.2, 0.3)

# --- Region grouping (simple, documented) ---------------------------------
# A coarse 6-region grouping of the Western US states present in the store
# (the speciation ingest is scoped to the Western bbox -125..-102 lon).
# Regions are kept intentionally coarse so the season_region features have a
# chance to generalize under leave-year-out; "other" exists for robustness
# (no binary sample currently maps there).
REGION_BY_STATE: Dict[str, str] = {
    "06": "california",                       # CA
    "04": "southwest", "32": "southwest", "35": "southwest",   # AZ, NV, NM
    "41": "pacific_northwest", "53": "pacific_northwest", "16": "pacific_northwest",  # OR, WA, ID
    "08": "mountain", "30": "mountain", "49": "mountain", "56": "mountain",  # CO, MT, UT, WY
    "31": "plains", "38": "plains", "46": "plains", "48": "plains",  # NE, ND, SD, TX
}
DEFAULT_REGION = "other"

# Season grouping of months (for the region-season holdout).
_SEASON_MONTHS = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}
_MONTH_TO_SEASON = {m: season for season, months in _SEASON_MONTHS.items() for m in months}

# --- Dust-opportunity (blowing-dust climatology) ----------------------------
# The Colorado Lamar rule: a day is a dust-opportunity when the daily max 10m
# wind gust >= 40 mph AND the antecedent 30-day precipitation (the sample day
# plus the 29 days before it) <= 0.6 in. ERA5 daily-max gusts are smoothed
# (a 10-minute mean over the hour max), so the rule is expected to fire
# rarely (low recall) but specifically (high precision) on IMPROVE dust days.
DUST_OPPORTUNITY_GUST_MIN = 40.0  # mph (daily max wind gust)
DUST_OPPORTUNITY_PRECIP_MAX = 0.6  # inches (antecedent 30-day precipitation)
PRECIP_30D_WINDOW_DAYS = 30  # sample day + 29 preceding days
MM_PER_INCH = 25.4


def region_for_state(state_code: str) -> str:
    """Region label for an AQS ``state_code`` (zero-padded FIPS two-digit)."""
    return REGION_BY_STATE.get(state_code, DEFAULT_REGION)


def season_of_month(month: int) -> str:
    """Season label ("DJF" | "MAM" | "JJA" | "SON") for a 1-based month."""
    return _MONTH_TO_SEASON.get(month, "DJF")


def transport_wind_features(
    u850: Optional[float], v850: Optional[float]
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Encode an 850 hPa wind vector into benchmark features.

    ``u850``/``v850`` are the eastward/northward components (m/s) from the
    ``transport_wind`` store. Returns ``(speed, dir_sin, dir_cos)`` where the
    direction is the METEOROLOGICAL direction the wind blows FROM, in degrees
    via ``mod(270 - degrees(atan2(v, u)), 360)`` (0° = from north, 90° = from
    east, 180° = from south, 270° = from west), encoded circularly as
    sin/cos so 359° and 1° are adjacent. Returns ``(None, None, None)`` when
    either component is missing (the ``t850_missing`` indicator covers that).
    """
    if u850 is None or v850 is None:
        return (None, None, None)
    speed = math.hypot(u850, v850)
    # atan2(v, u) is the direction the wind blows TOWARD (u eastward, v
    # northward); the direction it blows FROM is that rotated 270°, mod 360.
    dir_from_deg = math.fmod(270.0 - math.degrees(math.atan2(v850, u850)), 360.0)
    if dir_from_deg < 0.0:
        dir_from_deg += 360.0
    return (
        speed,
        math.sin(math.radians(dir_from_deg)),
        math.cos(math.radians(dir_from_deg)),
    )


def antecedent_precip_30d_in(
    precip_by_date: Dict[str, Optional[float]], date_local: str
) -> Tuple[Optional[float], bool]:
    """Antecedent 30-day precipitation for ``date_local``, in inches.

    ``precip_by_date`` maps ``YYYY-MM-DD`` -> daily precipitation in mm (the
    per-site cache from ``weather_daily.precipitation_mm``; a missing key is a
    day with no stored weather row). The window is the sample day PLUS the 29
    days before it (``PRECIP_30D_WINDOW_DAYS`` total), matching the Colorado
    Lamar rule ("30-day precipitation <= 0.6 in").

    Missing days and days whose precip value is None count as 0.0 for the sum
    but flip the returned ``missing`` flag so the modeling layer can impute.
    When NO day in the window has a known value, returns ``(None, True)`` (the
    sample's precip feature is entirely unknown and gets median-imputed).
    """
    from datetime import date, timedelta

    end = date.fromisoformat(date_local)
    total_mm = 0.0
    known_days = 0
    missing = False
    for offset in range(PRECIP_30D_WINDOW_DAYS):
        day = (end - timedelta(days=offset)).isoformat()
        value = precip_by_date.get(day)
        if value is None:
            missing = True
            continue
        total_mm += value
        known_days += 1
    if known_days == 0:
        return (None, True)
    return (total_mm / MM_PER_INCH, missing)


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpecBenchSample:
    """One binary smoke-vs-dust sample (composition answer key, PM-elevated).

    Features are the NON-CHEMICAL inference features only. ``wind_max_mph``,
    ``gust_max_mph``, ``pm25_pm10_ratio``, ``precip_30d_in``, and the 850 hPa
    transport-wind features (``t850_speed`` / ``t850_dir_sin`` /
    ``t850_dir_cos``) may be None when the source data is missing (the modeling
    layer imputes with the training-fold median and flags the ratio / gust /
    precip / transport-wind gaps via the ``ratio_missing`` / ``gust_missing`` /
    ``precip_missing`` / ``t850_missing`` indicators).
    """

    site_id: str
    date_local: str  # YYYY-MM-DD
    year: int
    month: int  # 1-12
    region: str
    spec_label: str  # "biomass_smoke" | "mineral_dust"
    pm_aqi: int  # max daily AQI over 88101/88502/81102
    wind_max_mph: Optional[float]
    pm25_pm10_ratio: Optional[float]
    gust_max_mph: Optional[float] = None  # daily max 10m wind gust (mph)
    precip_30d_in: Optional[float] = None  # antecedent 30-day precip sum (in)
    precip_missing: int = 0  # 1 when the 30-day precip window had gaps
    t850_speed: Optional[float] = None  # m/s at 850 hPa (NCEP Reanalysis-1)
    t850_dir_sin: Optional[float] = None  # circular encoding of direction FROM
    t850_dir_cos: Optional[float] = None


@dataclass
class SpecBenchDataset:
    """The assembled binary dataset plus the audit counts around it."""

    samples: List[SpecBenchSample]
    joinable_site_days: int  # speciation x aqs_daily site-days
    pm_elevated: int  # joinable site-days with max PM AQI > 50
    spec_distribution: Dict[str, int]  # composition class -> count (PM-elevated)
    per_year_spec: Dict[int, Dict[str, int]]  # year -> composition class -> count
    wind_imputed: int  # binary samples with no weather wind (imputed at fit)
    wind_total: int  # binary samples (for the imputation rate)
    ratio_missing: int  # binary samples with no PM2.5/PM10 ratio
    ratio_total: int  # binary samples (for the missing rate)
    gust_missing: int  # binary samples with no daily max gust
    gust_total: int  # binary samples (for the missing rate)
    precip_missing: int  # binary samples whose 30-day precip window is gappy
    precip_total: int  # binary samples (for the missing rate)
    t850_missing: int  # binary samples with no 850 hPa transport wind
    t850_total: int  # binary samples (for the missing rate)


def build_dataset(
    store: AccuracyStore,
    progress: Optional[Callable[[int, int], None]] = None,
) -> SpecBenchDataset:
    """Assemble the binary smoke-vs-dust dataset from the store.

    For every ``(site_id, date_local)`` joinable across ``speciation`` x
    ``aqs_daily``: keep it when the max PM AQI (88101/88502/81102) is > 50 and
    the IMPROVE-only composition label (``speclabels``) is a confident
    ``biomass_smoke`` or ``mineral_dust``. ``mixed`` / ``secondary_aerosol`` /
    ``ambiguous`` are counted (reported, excluded). Weather wind and the
    PM2.5/PM10 ratio are attached when present (None otherwise); imputation
    happens at fit time, and the assembly-level missing counts are logged.

    ``progress(i, total)`` fires once per joinable site-day (``total`` is the
    joinable count); the CLI passes a stderr printer.
    """
    site_days = store.fetch_aqs_speciation_site_days()
    total = len(site_days)

    samples: List[SpecBenchSample] = []
    spec_distribution: Dict[str, int] = {cls: 0 for cls in SPECIATION_CLASSES}
    per_year_spec: Dict[int, Dict[str, int]] = {}
    pm_elevated = 0
    wind_imputed = 0
    ratio_missing = 0
    gust_missing = 0
    precip_missing = 0
    t850_missing = 0

    # Per-site weather cache: weather_daily is keyed (site_id, date_local) and
    # only binary samples (a small fraction of the join) actually need it. The
    # cache carries the full WeatherDailyRecord so gust and the 30-day precip
    # window can be computed lazily; a parallel date->precip map avoids
    # re-deriving it per sample.
    weather_cache: Dict[str, Dict[str, WeatherDailyRecord]] = {}
    precip_cache: Dict[str, Dict[str, Optional[float]]] = {}
    # Same lazy pattern for the 850 hPa transport wind (transport_wind rows
    # carry raw u/v components; the derived features are computed below).
    transport_cache: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]] = {}

    for i, (site_id, date_local) in enumerate(site_days, start=1):
        if progress is not None:
            progress(i, total)
        year = int(date_local[:4])
        aqs_records = store.fetch_aqs_daily(site_id=site_id, date_local=date_local)
        pm_aqi = max(
            (r.aqi for r in aqs_records if r.parameter_code in PM_PARAM_CODES and r.aqi is not None),
            default=0,
        )
        if pm_aqi <= PM_AQI_ELEVATED:
            continue
        pm_elevated += 1

        components = derive_components(
            store.fetch_speciation(site_id, date_local),
            store.fetch_speciation_methods(site_id, date_local),
        )
        spec_label = classify_speciation(components, elevated=True)
        spec_distribution[spec_label] += 1
        per_year_spec.setdefault(year, {cls: 0 for cls in SPECIATION_CLASSES})
        per_year_spec[year][spec_label] += 1

        if spec_label not in (POSITIVE_CLASS, NEGATIVE_CLASS):
            continue

        # Weather wind / gust / antecedent precip (surface). None when the
        # weather row is missing.
        if site_id not in weather_cache:
            records = store.fetch_weather_daily(site_id=site_id)
            weather_cache[site_id] = {w.date_local: w for w in records}
            precip_cache[site_id] = {w.date_local: w.precipitation_mm for w in records}
        weather = weather_cache[site_id].get(date_local)
        wind_max_mph = weather.wind_max_mph if weather is not None else None
        if wind_max_mph is None:
            wind_imputed += 1
        gust_max_mph = weather.wind_gust_max_mph if weather is not None else None
        if gust_max_mph is None:
            gust_missing += 1
        precip_30d_in, precip_gap = antecedent_precip_30d_in(
            precip_cache[site_id], date_local
        )
        if precip_gap:
            precip_missing += 1

        # PM2.5/PM10 ratio via the same helper the production reconstruction
        # uses; None when only one of PM2.5/PM10 is present that day.
        ratio = _openaq_sig_from_aqs(aqs_records, date_local).get("pm25_pm10_ratio")
        if ratio is None:
            ratio_missing += 1

        # 850 hPa transport wind (NCEP/NCAR Reanalysis-1, 2.5°). The store
        # keeps the raw u/v components; the speed / circular direction
        # features are derived here. None when the site-day has no
        # transport_wind row (or the vector is incomplete).
        if site_id not in transport_cache:
            transport_cache[site_id] = {
                tw.date_local: (tw.u850, tw.v850)
                for tw in store.fetch_transport_wind(site_id=site_id)
            }
        u850, v850 = transport_cache[site_id].get(date_local, (None, None))
        t850_speed, t850_dir_sin, t850_dir_cos = transport_wind_features(u850, v850)
        if t850_speed is None:
            t850_missing += 1

        samples.append(
            SpecBenchSample(
                site_id=site_id,
                date_local=date_local,
                year=year,
                month=int(date_local[5:7]),
                region=region_for_state(site_id[:2]),
                spec_label=spec_label,
                pm_aqi=pm_aqi,
                wind_max_mph=wind_max_mph,
                pm25_pm10_ratio=ratio,
                gust_max_mph=gust_max_mph,
                precip_30d_in=precip_30d_in,
                precip_missing=1 if precip_gap else 0,
                t850_speed=t850_speed,
                t850_dir_sin=t850_dir_sin,
                t850_dir_cos=t850_dir_cos,
            )
        )

    return SpecBenchDataset(
        samples=samples,
        joinable_site_days=total,
        pm_elevated=pm_elevated,
        spec_distribution=spec_distribution,
        per_year_spec=per_year_spec,
        wind_imputed=wind_imputed,
        wind_total=len(samples),
        ratio_missing=ratio_missing,
        ratio_total=len(samples),
        gust_missing=gust_missing,
        gust_total=len(samples),
        precip_missing=precip_missing,
        precip_total=len(samples),
        t850_missing=t850_missing,
        t850_total=len(samples),
    )


# ---------------------------------------------------------------------------
# Features / ablations
# ---------------------------------------------------------------------------

# Continuous features (standardized per fold with training-fold stats) vs the
# binary indicators (kept 0/1). Region one-hot columns are named region_<name>.
_CONTINUOUS_FEATURES = (
    "pm_aqi",
    "wind_max_mph",
    "pm25_pm10_ratio",
    "month_sin",
    "month_cos",
    "gust_max_mph",
    "precip_30d_in",
    "t850_speed",
    "t850_dir_sin",
    "t850_dir_cos",
)
_BINARY_INDICATORS = (
    "ratio_missing",
    "t850_missing",
    "gust_missing",
    "precip_missing",
)

# The ablations, in display order.
ABLATION_ORDER: Tuple[str, ...] = (
    "severity_only",
    "wind_only",
    "ratio_only",
    "season_region_only",
    "t850_only",
    "dust_opportunity",
    "all",
    "all_plus_t850",
    "all_plus_dust",
    "all_minus_severity",
    "all_minus_season_region",
)


def make_feature_sets(regions: Sequence[str]) -> Dict[str, List[str]]:
    """Column list per ablation, given the region set present in the data.

    - ``severity_only``: [pm_aqi]
    - ``wind_only``: [wind_max_mph]
    - ``ratio_only``: [pm25_pm10_ratio, ratio_missing]
    - ``season_region_only``: [month_sin, month_cos, region_<r>...]
    - ``t850_only``: [t850_speed, t850_dir_sin, t850_dir_cos, t850_missing]
    - ``dust_opportunity``: [gust_max_mph, precip_30d_in, gust_missing,
      precip_missing] (the blowing-dust climatology block)
    - ``all``: everything above except the t850 and dust blocks
    - ``all_plus_t850``: ``all`` + the 850 hPa transport-wind features
    - ``all_plus_dust``: ``all`` + the dust-opportunity features
    - ``all_minus_severity``: all without pm_aqi
    - ``all_minus_season_region``: all without month/region
    """
    month_cols = ["month_sin", "month_cos"]
    region_cols = [f"region_{r}" for r in sorted(regions)]
    season_region = month_cols + region_cols
    severity = ["pm_aqi"]
    wind = ["wind_max_mph"]
    ratio = ["pm25_pm10_ratio", "ratio_missing"]
    t850 = ["t850_speed", "t850_dir_sin", "t850_dir_cos", "t850_missing"]
    dust = ["gust_max_mph", "precip_30d_in", "gust_missing", "precip_missing"]
    all_cols = severity + wind + ratio + season_region
    return {
        "severity_only": severity,
        "wind_only": wind,
        "ratio_only": ratio,
        "season_region_only": season_region,
        "t850_only": t850,
        "dust_opportunity": dust,
        "all": all_cols,
        "all_plus_t850": all_cols + t850,
        "all_plus_dust": all_cols + dust,
        "all_minus_severity": wind + ratio + season_region,
        "all_minus_season_region": severity + wind + ratio,
    }


def _regions(samples: Sequence[SpecBenchSample]) -> List[str]:
    """Distinct region labels present in the binary dataset (column basis)."""
    return sorted({s.region for s in samples})


def _raw_feature_row(sample: SpecBenchSample, regions: Sequence[str]) -> Dict[str, Optional[float]]:
    """One sample's raw feature dict. Wind/ratio/gust/precip/t850 are None when
    missing (the fit layer imputes them from the training fold); everything
    else is filled."""
    row: Dict[str, Optional[float]] = {
        "pm_aqi": float(sample.pm_aqi),
        "wind_max_mph": sample.wind_max_mph,
        "pm25_pm10_ratio": sample.pm25_pm10_ratio,
        "ratio_missing": 1.0 if sample.pm25_pm10_ratio is None else 0.0,
        "month_sin": math.sin(2.0 * math.pi * sample.month / 12.0),
        "month_cos": math.cos(2.0 * math.pi * sample.month / 12.0),
        "gust_max_mph": sample.gust_max_mph,
        "precip_30d_in": sample.precip_30d_in,
        "gust_missing": 1.0 if sample.gust_max_mph is None else 0.0,
        "precip_missing": float(sample.precip_missing),
        "t850_speed": sample.t850_speed,
        "t850_dir_sin": sample.t850_dir_sin,
        "t850_dir_cos": sample.t850_dir_cos,
        "t850_missing": 1.0 if sample.t850_speed is None else 0.0,
    }
    for region in regions:
        row[f"region_{region}"] = 1.0 if sample.region == region else 0.0
    return row


@dataclass
class _FeatureStats:
    """Fold-aware imputation + standardization stats (train-fold only)."""

    medians: Dict[str, float]  # wind / ratio medians for missing imputation
    mean: Dict[str, float]
    std: Dict[str, float]


def _fit_feature_stats(
    train_raw: Sequence[Dict[str, Optional[float]]], feature_cols: Sequence[str]
) -> _FeatureStats:
    """Training-fold stats: median imputation for wind/ratio/t850, mean/std for
    the continuous features. Never touches the test fold."""
    medians: Dict[str, float] = {}
    for col in (
        "wind_max_mph",
        "pm25_pm10_ratio",
        "gust_max_mph",
        "precip_30d_in",
        "t850_speed",
        "t850_dir_sin",
        "t850_dir_cos",
    ):
        vals = [row[col] for row in train_raw if row[col] is not None]
        medians[col] = _median(vals) if vals else 0.0

    mean: Dict[str, float] = {}
    std: Dict[str, float] = {}
    for col in feature_cols:
        if col in _BINARY_INDICATORS or col.startswith("region_"):
            continue
        vals = [row[col] if row[col] is not None else medians.get(col, 0.0) for row in train_raw]
        m = sum(vals) / len(vals)
        s = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
        mean[col] = m
        std[col] = s if s > 1e-12 else 1.0
    return _FeatureStats(medians=medians, mean=mean, std=std)


def _encode(
    raw_rows: Sequence[Dict[str, Optional[float]]],
    feature_cols: Sequence[str],
    stats: _FeatureStats,
) -> List[List[float]]:
    """Encode raw feature dicts into the standardized design matrix (imputing
    missing wind/ratio with the training-fold median)."""
    matrix = []
    for row in raw_rows:
        vec = []
        for col in feature_cols:
            if col in _BINARY_INDICATORS or col.startswith("region_"):
                vec.append(float(row[col]))
            else:
                value = row[col] if row[col] is not None else stats.medians.get(col, 0.0)
                vec.append((value - stats.mean[col]) / stats.std[col])
        matrix.append(vec)
    return matrix


# ---------------------------------------------------------------------------
# L2 logistic regression (pure stdlib; small matrices -> Newton-IRLS)
# ---------------------------------------------------------------------------


def _sigmoid(z: float) -> float:
    if z < -40.0:
        return 1e-7
    if z > 40.0:
        return 1.0 - 1e-7
    return 1.0 / (1.0 + 2.718281828459045 ** -z)


def _mat_vec(matrix: List[List[float]], vec: List[float]) -> List[float]:
    return [sum(a * v for a, v in zip(row, vec)) for row in matrix]


def _trans_mat_vec(matrix: List[List[float]], vec: List[float]) -> List[float]:
    """``matrix.T @ vec`` (matrix is a Python list of rows)."""
    cols = len(matrix[0]) if matrix else 0
    return [sum(matrix[i][j] * vec[i] for i in range(len(matrix))) for j in range(cols)]


def _solve_linear(aug: List[List[float]]) -> List[float]:
    """Gaussian elimination with partial pivoting on the augmented matrix."""
    n = len(aug)
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        pv = aug[col][col]
        if abs(pv) < 1e-12:
            continue
        for r in range(col + 1, n):
            factor = aug[r][col] / pv
            for c in range(col, n + 1):
                aug[r][c] -= factor * aug[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = aug[r][n] - sum(aug[r][c] * x[c] for c in range(r + 1, n))
        x[r] = s / aug[r][r]
    return x


class LogisticL2:
    """L2-regularized logistic regression fit by Newton-IRLS (no scipy).

    The intercept column is not regularized; all predictor coefficients are
    penalized by ``l2`` (default 1.0, matching sklearn's LogisticRegression
    default scale). A degenerate training fold with a single class is handled
    by predicting that class (balanced accuracy is then ill-defined on a
    two-class test set, so callers must guard with the fold's class counts).
    """

    def __init__(self, l2: float = 1.0, max_iter: int = 40, tol: float = 1e-8):
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol
        self.beta: Optional[List[float]] = None
        self.single_class: Optional[int] = None

    def fit(self, X: List[List[float]], y: List[int]) -> "LogisticL2":
        classes = list(set(y))
        if len(classes) < 2:
            self.single_class = 1 if classes[0] == 1 else 0
            self.beta = None
            return self
        self.single_class = None

        Xw = [[1.0] + row for row in X]
        n, k = len(Xw), len(Xw[0])
        beta = [0.0] * k
        reg = [0.0] + [self.l2] * (k - 1)

        for _ in range(self.max_iter):
            eta = _mat_vec(Xw, beta)
            mu = [_sigmoid(e) for e in eta]
            weights = [max(m * (1.0 - m), 1e-12) for m in mu]
            z = [
                e + (y_i - m) / w
                for e, m, w, y_i in zip(eta, mu, weights, y)
            ]
            # H = Xw.T @ (Xw * w) + diag(reg)
            H = []
            for j in range(k):
                Hj = []
                for ell in range(k):
                    s = sum(Xw[i][j] * Xw[i][ell] * weights[i] for i in range(n))
                    Hj.append(s + (reg[j] if j == ell else 0.0))
                H.append(Hj)
            rhs = _trans_mat_vec(Xw, [w * zz for w, zz in zip(weights, z)])
            aug = [H[j] + [rhs[j]] for j in range(k)]
            beta_new = _solve_linear(aug)
            if max(abs(a - b) for a, b in zip(beta, beta_new)) < self.tol:
                beta = beta_new
                break
            beta = beta_new
        self.beta = beta
        return self

    def predict_proba(self, X: List[List[float]]) -> List[float]:
        if self.single_class is not None:
            return [1.0 if self.single_class == 1 else 0.0] * len(X)
        Xw = [[1.0] + row for row in X]
        return [_sigmoid(e) for e in _mat_vec(Xw, self.beta)]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _median(vals: Sequence[float]) -> float:
    ordered = sorted(vals)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _balanced_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> Optional[float]:
    """Mean per-class recall. None when the test set lacks a class."""
    classes = sorted(set(y_true))
    if len(classes) < 2:
        return None
    recalls = []
    for c in classes:
        mask = [i for i, v in enumerate(y_true) if v == c]
        recalls.append(sum(y_pred[i] == c for i in mask) / len(mask))
    return sum(recalls) / len(recalls)


def _roc_auc(y_true: Sequence[int], score: Sequence[float]) -> Optional[float]:
    """Rank-free Mann-Whitney AUC over observed pairs. None without both
    classes. Small enough here (a few hundred test samples) to do pairwise."""
    pos = [s for s, y in zip(score, y_true) if y == 1]
    neg = [s for s, y in zip(score, y_true) if y == 0]
    if not pos or not neg:
        return None
    greater = equal = 0
    for p in pos:
        for n_val in neg:
            if p > n_val:
                greater += 1
            elif p == n_val:
                equal += 1
    return (greater + 0.5 * equal) / (len(pos) * len(neg))


def _label_vector(samples: Sequence[SpecBenchSample]) -> List[int]:
    return [1 if s.spec_label == POSITIVE_CLASS else 0 for s in samples]


def _fit_and_predict(
    train: Sequence[SpecBenchSample],
    test: Sequence[SpecBenchSample],
    feature_cols: Sequence[str],
    regions: Sequence[str],
) -> Tuple[LogisticL2, List[float], List[int]]:
    """Fit L2 logistic on ``train`` (fold-aware imputation/standardization)
    and return ``(model, proba_on_test, y_test)``."""
    train_raw = [_raw_feature_row(s, regions) for s in train]
    test_raw = [_raw_feature_row(s, regions) for s in test]
    stats = _fit_feature_stats(train_raw, feature_cols)
    X_train = _encode(train_raw, feature_cols, stats)
    X_test = _encode(test_raw, feature_cols, stats)
    model = LogisticL2()
    model.fit(X_train, _label_vector(train))
    return model, model.predict_proba(X_test), _label_vector(test)


# ---------------------------------------------------------------------------
# Split regimes
# ---------------------------------------------------------------------------


def _eval_ablation(
    train: Sequence[SpecBenchSample],
    test: Sequence[SpecBenchSample],
    feature_cols: Sequence[str],
    regions: Sequence[str],
) -> Dict[str, object]:
    """Fit one ablation on train, evaluate on test (balanced acc + AUC)."""
    model, proba, y_test = _fit_and_predict(train, test, feature_cols, regions)
    pred = [1 if p >= 0.5 else 0 for p in proba]
    return {
        "n_test": len(test),
        "n_smoke": sum(1 for y in y_test if y == 1),
        "n_dust": sum(1 for y in y_test if y == 0),
        "balanced_accuracy": _balanced_accuracy(y_test, pred),
        "auc": _roc_auc(y_test, proba),
        "mean_margin": (
            float(sum(abs(p - 0.5) for p in proba) / len(proba)) if proba else 0.0
        ),
    }


def leave_year_out(
    dataset: SpecBenchDataset,
    feature_sets: Dict[str, List[str]],
    years: Sequence[int] = BENCHMARK_YEARS,
) -> Dict[str, List[Dict[str, object]]]:
    """Train on all years except Y, evaluate on Y, for each Y. Per ablation
    returns one result dict per year (with ``year`` attached)."""
    samples = dataset.samples
    regions = _regions(samples)
    results: Dict[str, List[Dict[str, object]]] = {}
    for name, cols in feature_sets.items():
        folds: List[Dict[str, object]] = []
        for year in years:
            train = [s for s in samples if s.year != year]
            test = [s for s in samples if s.year == year]
            fold = _eval_ablation(train, test, cols, regions)
            fold["year"] = year
            folds.append(fold)
        results[name] = folds
    return results


def pick_dust_heaviest_region_season(
    samples: Sequence[SpecBenchSample],
) -> Tuple[Tuple[str, str], int, int]:
    """The ``(region, season)`` with the most mineral-dust samples (ties: the
    higher dust share), plus its ``(dust, smoke)`` counts — the documented
    choice for the region-season holdout (the "Kansas wind" concern)."""
    counts: Dict[Tuple[str, str], List[int]] = {}
    for s in samples:
        key = (s.region, season_of_month(s.month))
        bucket = counts.setdefault(key, [0, 0])
        bucket[0 if s.spec_label == NEGATIVE_CLASS else 1] += 1

    def rank(key: Tuple[str, str]) -> Tuple[int, float]:
        dust, smoke = counts[key]
        share = dust / (dust + smoke) if (dust + smoke) else 0.0
        return (dust, share)

    best = max(counts, key=rank)
    dust, smoke = counts[best]
    return best, dust, smoke


def region_season_holdout(
    dataset: SpecBenchDataset,
    feature_sets: Dict[str, List[str]],
    holdout: Tuple[str, str],
) -> Dict[str, Dict[str, object]]:
    """Train on every sample outside one ``(region, season)``, evaluate on the
    holdout. Directly tests the region-season confound: can a model trained
    elsewhere tell smoke from dust in the dust-heaviest region-season?"""
    samples = dataset.samples
    regions = _regions(samples)
    holdout_region, holdout_season = holdout

    def in_holdout(s: SpecBenchSample) -> bool:
        return s.region == holdout_region and season_of_month(s.month) == holdout_season

    train = [s for s in samples if not in_holdout(s)]
    test = [s for s in samples if in_holdout(s)]
    results: Dict[str, Dict[str, object]] = {}
    for name, cols in feature_sets.items():
        results[name] = _eval_ablation(train, test, cols, regions)
    return results


def severity_strata(
    dataset: SpecBenchDataset,
    feature_sets: Dict[str, List[str]],
    years: Sequence[int] = BENCHMARK_YEARS,
    model_names: Sequence[str] = ("severity_only", "all"),
) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Per-stratum balanced accuracy when the evaluation is split into pm_aqi
    terciles cut on the TRAINING fold (no leakage). Returns
    ``model_name -> stratum -> {balanced_accuracy, n, n_smoke, n_dust}``
    averaged over leave-year-out folds (per-stratum values, one entry per
    stratum present across folds)."""
    samples = dataset.samples
    regions = _regions(samples)
    # stratum -> list of per-fold balanced accuracies (per model)
    strata: Dict[str, Dict[str, List[Optional[float]]]] = {
        name: {"low": [], "mid": [], "high": []} for name in model_names
    }
    strata_n: Dict[str, Dict[str, Dict[str, int]]] = {
        name: {s: {"n": 0, "n_smoke": 0, "n_dust": 0} for s in ("low", "mid", "high")}
        for name in model_names
    }

    for year in years:
        train = [s for s in samples if s.year != year]
        test = [s for s in samples if s.year == year]
        if not test:
            continue
        train_aqis = sorted(s.pm_aqi for s in train)
        low_cut = _quantile(train_aqis, 1 / 3)
        high_cut = _quantile(train_aqis, 2 / 3)

        for name in model_names:
            model, proba, y_test = _fit_and_predict(train, test, feature_sets[name], regions)
            pred = [1 if p >= 0.5 else 0 for p in proba]
            # Per-stratum balanced accuracy for this fold. Strata are binned by
            # the TRAINING fold's pm_aqi terciles (no leakage into the cut).
            for stratum, (lo, hi) in (
                ("low", (-float("inf"), low_cut)),
                ("mid", (low_cut, high_cut)),
                ("high", (high_cut, float("inf"))),
            ):
                idx = [
                    i for i, s in enumerate(test) if lo < s.pm_aqi <= hi
                ]
                fold_y = [y_test[i] for i in idx]
                fold_pred = [pred[i] for i in idx]
                strata_n[name][stratum]["n"] += len(idx)
                strata_n[name][stratum]["n_smoke"] += sum(1 for y in fold_y if y == 1)
                strata_n[name][stratum]["n_dust"] += sum(1 for y in fold_y if y == 0)
                strata[name][stratum].append(_balanced_accuracy(fold_y, fold_pred))

    out: Dict[str, Dict[str, Dict[str, object]]] = {}
    for name in model_names:
        out[name] = {}
        for stratum in ("low", "mid", "high"):
            bal = [b for b in strata[name][stratum] if b is not None]
            out[name][stratum] = {
                "balanced_accuracy": _mean(bal) if bal else None,
                "folds": len(bal),
                "n": strata_n[name][stratum]["n"],
                "n_smoke": strata_n[name][stratum]["n_smoke"],
                "n_dust": strata_n[name][stratum]["n_dust"],
            }
    return out


def severity_matched_subset(
    samples: Sequence[SpecBenchSample], tolerance: float = SEVERITY_MATCH_TOL
) -> List[Tuple[SpecBenchSample, SpecBenchSample]]:
    """Greedy severity-matched pairs: for each smoke sample (by pm_aqi) find
    the nearest unmatched dust sample within ``tolerance`` AQI points. The
    cleanest "is this source signal or a severity shortcut" test set."""
    smoke = sorted(
        [s for s in samples if s.spec_label == POSITIVE_CLASS], key=lambda s: s.pm_aqi
    )
    dust = sorted(
        [s for s in samples if s.spec_label == NEGATIVE_CLASS], key=lambda s: s.pm_aqi
    )
    used: set = set()
    pairs: List[Tuple[SpecBenchSample, SpecBenchSample]] = []
    for s in smoke:
        best: Optional[Tuple[float, int]] = None
        for i, d in enumerate(dust):
            if i in used:
                continue
            diff = abs(d.pm_aqi - s.pm_aqi)
            if diff <= tolerance and (best is None or diff < best[0]):
                best = (diff, i)
        if best is not None:
            used.add(best[1])
            pairs.append((s, dust[best[1]]))
    return pairs


def severity_matched_eval(
    dataset: SpecBenchDataset,
    feature_sets: Dict[str, List[str]],
    years: Sequence[int] = BENCHMARK_YEARS,
    tolerance: float = SEVERITY_MATCH_TOL,
) -> Dict[str, object]:
    """Leave-year-out where the EVALUATION is restricted to the severity-
    matched subset of each fold's test year (train stays the full train fold).
    If performance collapses vs the full fold, the model leaned on severity;
    if it survives, there is residual source signal."""
    samples = dataset.samples
    regions = _regions(samples)
    per_ablation: Dict[str, List[Optional[float]]] = {name: [] for name in feature_sets}
    per_ablation_n: Dict[str, int] = {name: 0 for name in feature_sets}
    per_ablation_full: Dict[str, List[Optional[float]]] = {name: [] for name in feature_sets}
    matched_total = 0

    for year in years:
        train = [s for s in samples if s.year != year]
        test = [s for s in samples if s.year == year]
        pairs = severity_matched_subset(test, tolerance=tolerance)
        matched_total += len(pairs)
        if not pairs:
            for name in feature_sets:
                per_ablation[name].append(None)
                per_ablation_full[name].append(None)
            continue
        matched_test = [d for pair in pairs for d in pair]  # smoke then dust, interleaved
        for name, cols in feature_sets.items():
            model, proba, y_test = _fit_and_predict(train, matched_test, cols, regions)
            pred = [1 if p >= 0.5 else 0 for p in proba]
            per_ablation[name].append(_balanced_accuracy(y_test, pred))
            per_ablation_full[name].append(
                _eval_ablation(train, test, cols, regions)["balanced_accuracy"]
            )
            per_ablation_n[name] += len(matched_test)

    return {
        "tolerance": tolerance,
        "matched_pairs_total": matched_total,
        "per_ablation": {
            name: {
                "matched_balanced_accuracy": _mean_std([b for b in per_ablation[name] if b is not None]),
                "matched_folds": sum(1 for b in per_ablation[name] if b is not None),
                "full_balanced_accuracy": _mean_std(
                    [b for b in per_ablation_full[name] if b is not None]
                ),
                "matched_n": per_ablation_n[name],
            }
            for name in feature_sets
        },
    }


def abstention_curve(
    dataset: SpecBenchDataset,
    feature_sets: Dict[str, List[str]],
    years: Sequence[int] = BENCHMARK_YEARS,
    taus: Sequence[float] = ABSTENTION_TAUS,
) -> Dict[str, Dict[str, object]]:
    """For the ``all`` model under leave-year-out: pool every fold's test
    samples with their P(smoke), then report coverage and balanced accuracy on
    the subset with |P(smoke) - 0.5| >= tau. Shows whether the model can
    honestly abstain (predict only the samples it is confident about)."""
    samples = dataset.samples
    regions = _regions(samples)
    all_cols = feature_sets["all"]
    pooled_y: List[int] = []
    pooled_proba: List[float] = []
    for year in years:
        train = [s for s in samples if s.year != year]
        test = [s for s in samples if s.year == year]
        if not test:
            continue
        _, proba, y_test = _fit_and_predict(train, test, all_cols, regions)
        pooled_y.extend(y_test)
        pooled_proba.extend(proba)

    curve: Dict[str, Dict[str, object]] = {}
    for tau in taus:
        idx = [i for i, p in enumerate(pooled_proba) if abs(p - 0.5) >= tau]
        sub_y = [pooled_y[i] for i in idx]
        sub_proba = [pooled_proba[i] for i in idx]
        sub_pred = [1 if p >= 0.5 else 0 for p in sub_proba]
        curve[f"tau_{tau}"] = {
            "tau": tau,
            "coverage": len(idx) / len(pooled_y) if pooled_y else None,
            "n": len(idx),
            "n_smoke": sum(1 for y in sub_y if y == 1),
            "n_dust": sum(1 for y in sub_y if y == 0),
            "balanced_accuracy": _balanced_accuracy(sub_y, sub_pred),
            "auc": _roc_auc(sub_y, sub_proba),
        }
    return curve


# ---------------------------------------------------------------------------
# Dust-opportunity hard rule (no model; the literal Lamar rule)
# ---------------------------------------------------------------------------


def dust_opportunity_rule_prediction(sample: SpecBenchSample) -> Optional[bool]:
    """Classify one sample by the literal Lamar dust rule.

    Returns True (predict dust) when ``gust_max_mph >= 40`` AND the antecedent
    ``precip_30d_in <= 0.6``; False (predict smoke) otherwise. Returns None
    when gust or precip is missing — the rule cannot apply, and callers treat
    None as "not dust" (the rule's ``else smoke`` branch) while reporting it
    separately as unclassified (coverage).
    """
    if sample.gust_max_mph is None or sample.precip_30d_in is None:
        return None
    return (
        sample.gust_max_mph >= DUST_OPPORTUNITY_GUST_MIN
        and sample.precip_30d_in <= DUST_OPPORTUNITY_PRECIP_MAX
    )


def _dust_rule_stats(samples: Sequence[SpecBenchSample]) -> Dict[str, object]:
    """Precision / recall / coverage of the Lamar rule over ``samples``.

    ``mineral_dust`` is the positive class. Samples missing gust or precip are
    treated as "not dust" for precision/recall (the rule's else-smoke branch)
    and counted as unclassified, so ``coverage`` = fraction the rule was
    defined on.
    """
    n = len(samples)
    pred_dust = 0
    actual_dust = 0
    tp = 0
    unclassified = 0
    for s in samples:
        is_dust = s.spec_label == NEGATIVE_CLASS
        if is_dust:
            actual_dust += 1
        pred = dust_opportunity_rule_prediction(s)
        if pred is None:
            unclassified += 1
            continue
        if pred:
            pred_dust += 1
            if is_dust:
                tp += 1
    fp = pred_dust - tp
    fn = actual_dust - tp
    return {
        "n": n,
        "classified": n - unclassified,
        "coverage": (n - unclassified) / n if n else None,
        "pred_dust": pred_dust,
        "actual_dust": actual_dust,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": (tp / (tp + fp)) if (tp + fp) else None,
        "recall": (tp / (tp + fn)) if (tp + fn) else None,
    }


def dust_opportunity_rule_diagnostic(dataset: SpecBenchDataset) -> Dict[str, object]:
    """The Lamar-rule diagnostic: precision / recall / coverage over the full
    binary set and per year. No model is fit; ERA5 gusts are smoothed, so we
    expect high precision, low recall."""
    samples = dataset.samples
    return {
        "full": _dust_rule_stats(samples),
        "per_year": {
            year: _dust_rule_stats([s for s in samples if s.year == year])
            for year in sorted({s.year for s in samples})
        },
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _mean(vals: Sequence[float]) -> float:
    return sum(vals) / len(vals)


def _mean_std(vals: Sequence[float]) -> Optional[Tuple[float, float]]:
    if not vals:
        return None
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return (m, 0.0)
    s = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    return (m, s)


def _quantile(sorted_vals: List[float], q: float) -> float:
    """Linear-interpolation quantile over an ascending list (numpy-compatible
    for the p=1/3, 2/3 case used here)."""
    if not sorted_vals:
        return 0.0
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _fmt_pct(value: Optional[float], width: int = 7) -> str:
    """Format a balanced-accuracy fraction as a percentage string (``-`` for
    None, e.g. an absent class in the fold)."""
    if value is None:
        return f"{'-':>{width}}"
    return f"{value * 100:>{width}.1f}"


def _fmt_ms(value: Optional[Tuple[float, float]], width: int = 7) -> str:
    if value is None:
        return f"{'-':>{width}}"
    return f"{value[0] * 100:>{width}.1f}±{value[1] * 100:.1f}"


# ---------------------------------------------------------------------------
# Benchmark runner + report
# ---------------------------------------------------------------------------


def run_specbench(
    store: AccuracyStore,
    years: Sequence[int] = BENCHMARK_YEARS,
    tolerance: float = SEVERITY_MATCH_TOL,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, object]:
    """Run the full multi-year benchmark over the store and return the result
    dict consumed by ``format_report``."""
    dataset = build_dataset(store, progress=progress)
    regions = _regions(dataset.samples)
    feature_sets = make_feature_sets(regions)

    holdout_key, holdout_dust, holdout_smoke = pick_dust_heaviest_region_season(
        dataset.samples
    )
    matched = severity_matched_subset(dataset.samples, tolerance=tolerance)

    return {
        "dataset": dataset,
        "feature_sets": feature_sets,
        "holdout_region_season": holdout_key,
        "holdout_counts": {"dust": holdout_dust, "smoke": holdout_smoke},
        "matched_pairs": matched,
        "leave_year_out": leave_year_out(dataset, feature_sets, years=years),
        "region_season_holdout": region_season_holdout(dataset, feature_sets, holdout_key),
        "severity_strata": severity_strata(dataset, feature_sets, years=years),
        "severity_matched": severity_matched_eval(
            dataset, feature_sets, years=years, tolerance=tolerance
        ),
        "abstention": abstention_curve(dataset, feature_sets, years=years),
        "dust_rule": dust_opportunity_rule_diagnostic(dataset),
    }


def format_report(result: Dict[str, object]) -> str:
    """Render the full benchmark report as text (the ``specbench`` CLI output)."""
    dataset: SpecBenchDataset = result["dataset"]
    feature_sets: Dict[str, List[str]] = result["feature_sets"]
    lines: List[str] = []
    add = lines.append

    add("=" * 78)
    add("MULTI-YEAR SMOKE-vs-DUST ABLATION BENCHMARK (severity vs source signal)")
    add("=" * 78)
    add(
        "Ground truth: speclabels COMPOSITION answer key (IMPROVE method filter only).\n"
        "Features: non-chemical inference features only (severity, surface wind,\n"
        "gust, antecedent 30-day precip, PM2.5/PM10 ratio, season, region,\n"
        "850 hPa transport wind). Fire/HMS features are NOT included: multi-year\n"
        "FIRMS/HMS is not ingested yet.\n"
        "Production scorer and Params untouched."
    )

    # -- dataset ------------------------------------------------------------
    add("\n" + "=" * 78)
    add("1. DATASET ASSEMBLY (speciation x aqs_daily, PM-elevated)")
    add("=" * 78)
    add(
        f"joinable site-days (speciation x aqs_daily): {dataset.joinable_site_days}\n"
        f"PM-elevated (max AQI over 88101/88502/81102 > {PM_AQI_ELEVATED}): {dataset.pm_elevated}"
    )
    add("\ncomposition distribution over PM-elevated days (answer key):")
    for cls in SPECIATION_CLASSES:
        add(f"  {cls:<18} {dataset.spec_distribution.get(cls, 0)}")
    add(
        f"\nbinary dataset (biomass_smoke=1 vs mineral_dust=0): {len(dataset.samples)}\n"
        f"  smoke:  {sum(1 for s in dataset.samples if s.spec_label == POSITIVE_CLASS)}\n"
        f"  dust:   {sum(1 for s in dataset.samples if s.spec_label == NEGATIVE_CLASS)}"
    )
    add("\nper-year PM-elevated composition counts:")
    add(f"  {'year':<6}{'smoke':>8}{'dust':>8}{'mixed':>8}{'second':>8}{'ambig':>8}{'total':>8}")
    for year in sorted(dataset.per_year_spec):
        row = dataset.per_year_spec[year]
        total = sum(row.values())
        add(
            f"  {year:<6}{row['biomass_smoke']:>8}{row['mineral_dust']:>8}"
            f"{row['mixed']:>8}{row['secondary_aerosol']:>8}{row['ambiguous']:>8}{total:>8}"
        )
    binary_per_year = {y: {"smoke": 0, "dust": 0} for y in dataset.per_year_spec}
    for s in dataset.samples:
        bucket = binary_per_year.setdefault(s.year, {"smoke": 0, "dust": 0})
        bucket["smoke" if s.spec_label == POSITIVE_CLASS else "dust"] += 1
    add("\nbinary dataset per year:")
    add(f"  {'year':<6}{'smoke':>8}{'dust':>8}{'total':>8}")
    for year in sorted(binary_per_year):
        row = binary_per_year[year]
        add(f"  {year:<6}{row['smoke']:>8}{row['dust']:>8}{row['smoke'] + row['dust']:>8}")

    add(
        f"\nmissing-data audit (binary set):\n"
        f"  wind_max_mph imputed (no weather wind): {dataset.wind_imputed}/{dataset.wind_total}\n"
        f"  gust_max_mph missing (gust_missing=1): {dataset.gust_missing}/{dataset.gust_total}\n"
        f"  antecedent 30-day precip gappy (precip_missing=1): {dataset.precip_missing}/{dataset.precip_total}\n"
        f"  pm25_pm10_ratio missing (ratio_missing=1): {dataset.ratio_missing}/{dataset.ratio_total}\n"
        f"  850 hPa transport wind missing (t850_missing=1): {dataset.t850_missing}/{dataset.t850_total}"
    )

    # -- ablations ----------------------------------------------------------
    add("\n" + "=" * 78)
    add("2. ABLATIONS (all L2 logistic, standardized, binary smoke=1/dust=0)")
    add("=" * 78)
    for name in ABLATION_ORDER:
        add(f"  {name:<24} {', '.join(feature_sets[name])}")

    # -- leave-year-out -----------------------------------------------------
    add("\n" + "=" * 78)
    add("3. LEAVE-YEAR-OUT (train all years except Y, evaluate on Y)")
    add("=" * 78)
    lyo: Dict[str, List[Dict[str, object]]] = result["leave_year_out"]
    fold_years = sorted(f["year"] for f in lyo["all"])
    header = f"  {'ablation':<24}" + "".join(f"{y:>9}" for y in fold_years) + f"{'mean':>10}"
    add(header)
    add("  " + "-" * (len(header) - 2))
    for name in ABLATION_ORDER:
        folds = sorted(lyo[name], key=lambda f: f["year"])
        vals = [f["balanced_accuracy"] for f in folds if f["balanced_accuracy"] is not None]
        ms = _mean_std(vals)
        cells = "".join(_fmt_pct(f["balanced_accuracy"], width=9) for f in folds)
        mean_cell = f"{'-':>10}" if ms is None else f"{ms[0] * 100:>9.1f}±{ms[1] * 100:.1f}"
        add(f"  {name:<24}{cells}{mean_cell}")
    add("\nAUC per fold (threshold-free complement):")
    for name in ABLATION_ORDER:
        folds = sorted(lyo[name], key=lambda f: f["year"])
        cells = "".join(
            ("-" if f["auc"] is None else f"{f['auc']:.3f}").rjust(9)
            for f in folds
        )
        add(f"  {name:<24}{cells}")

    # -- region-season holdout ----------------------------------------------
    add("\n" + "=" * 78)
    add("4. REGION-SEASON HOLDOUT (the 'Kansas wind' confound test)")
    add("=" * 78)
    (holdout_region, holdout_season), counts = (
        result["holdout_region_season"],
        result["holdout_counts"],
    )
    add(
        f"held out: {holdout_region} / {holdout_season} "
        f"(dust-heaviest region-season in the data: {counts['dust']} dust, "
        f"{counts['smoke']} smoke)"
    )
    rs = result["region_season_holdout"]
    for name in ABLATION_ORDER:
        r = rs[name]
        bal = r["balanced_accuracy"]
        auc = r["auc"]
        add(
            f"  {name:<24} bal_acc={_fmt_pct(bal)}  auc={'-' if auc is None else f'{auc:.3f}'}  "
            f"n={r['n_test']} (smoke {r['n_smoke']}, dust {r['n_dust']})"
        )

    # -- severity strata ----------------------------------------------------
    add("\n" + "=" * 78)
    add("5. SEVERITY STRATA (pm_aqi terciles cut on the train fold; eval binned)")
    add("=" * 78)
    strata: Dict[str, Dict[str, Dict[str, object]]] = result["severity_strata"]
    add(
        f"  {'model':<24}{'low tertile':>14}{'mid tertile':>14}{'high tertile':>14}"
    )
    for name in ("severity_only", "all"):
        row = strata[name]
        add(
            f"  {name:<24}"
            + "".join(
                _fmt_pct(row[stratum]["balanced_accuracy"], width=14)
                for stratum in ("low", "mid", "high")
            )
        )
    add("  per-stratum sample sizes (all model):")
    row = strata["all"]
    add(
        f"  {'':<24}"
        + "".join(f"{row[s]['n']:>14}" for s in ("low", "mid", "high"))
    )
    add(
        f"  {'':<24}"
        + "".join(
            f"({row[s]['n_smoke']}s/{row[s]['n_dust']}d)".rjust(14) for s in ("low", "mid", "high")
        )
    )

    # -- severity-matched subset --------------------------------------------
    add("\n" + "=" * 78)
    add("6. SEVERITY-MATCHED SUBSET (nearest-severity dust per smoke)")
    add("=" * 78)
    matched = result["severity_matched"]
    add(
        f"greedy match tolerance: |pm_aqi_dust - pm_aqi_smoke| <= {matched['tolerance']:.0f}\n"
        f"matched smoke-dust pairs (all years): {matched['matched_pairs_total']}"
    )
    add(
        f"  {'ablation':<24}{'matched bal_acc':>18}{'full-fold bal_acc':>20}"
    )
    for name in ABLATION_ORDER:
        row = matched["per_ablation"][name]
        m = row["matched_balanced_accuracy"]
        f = row["full_balanced_accuracy"]
        add(
            f"  {name:<24}{_fmt_ms(m, width=18)}{_fmt_ms(f, width=20)}"
        )

    # -- dust-opportunity hard rule -----------------------------------------
    add("\n" + "=" * 78)
    add("7. DUST-OPPORTUNITY HARD RULE (no model; the literal Lamar rule)")
    add("=" * 78)
    add(
        "classify dust when gust_max_mph >= 40 AND antecedent 30-day precip\n"
        "<= 0.6 in, else smoke (missing gust/precip = not dust, counted as\n"
        "unclassified). ERA5 daily-max gusts are smoothed, so expect high\n"
        "precision, low recall, and partial coverage."
    )
    rule = result["dust_rule"]

    def _rule_row(label: str, stats: Dict[str, object]) -> None:
        prec = stats["precision"]
        rec = stats["recall"]
        cov = stats["coverage"]
        add(
            f"  {label:<10} n={stats['n']:>5} classified={stats['classified']:>5}"
            f" coverage={0.0 if cov is None else cov * 100:>5.1f}%"
            f" precision={'-' if prec is None else f'{prec * 100:.1f}%':>7}"
            f" recall={'-' if rec is None else f'{rec * 100:.1f}%':>7}"
            f" pred_dust={stats['pred_dust']:>4}"
            f" (tp {stats['tp']}, fp {stats['fp']})"
        )

    _rule_row("full", rule["full"])
    for year, stats in sorted(rule["per_year"].items()):
        _rule_row(str(year), stats)

    # -- abstention ---------------------------------------------------------
    add("\n" + "=" * 78)
    add("8. ABSTENTION CURVE (all model; predict only when |P(smoke)-0.5| >= tau)")
    add("=" * 78)
    add("note: pools every leave-year-out test sample across folds (coverage-")
    add("weighted); tau=0.0 is therefore the pooled all-model balanced accuracy,")
    add("not the mean of the per-fold values in section 3.")
    add(f"  {'tau':<8}{'coverage':>10}{'bal_acc':>10}{'auc':>10}{'n':>8}{'smoke':>8}{'dust':>8}")
    for row in result["abstention"].values():
        auc_cell = "-" if row["auc"] is None else f"{row['auc']:.3f}"
        add(
            f"  {row['tau']:<8.1f}"
            f"{(0.0 if row['coverage'] is None else row['coverage'] * 100):>9.1f}%"
            f"{_fmt_pct(row['balanced_accuracy'], width=10)}"
            f"{auc_cell.rjust(10)}"
            f"{row['n']:>8}{row['n_smoke']:>8}{row['n_dust']:>8}"
        )

    # -- findings -----------------------------------------------------------
    add("\n" + "=" * 78)
    add("9. FINDINGS")
    add("=" * 78)
    lyo_all = lyo["all"]
    lyo_sev = lyo["severity_only"]
    lyo_ams = lyo["all_minus_severity"]
    mean_all = _mean_std([f["balanced_accuracy"] for f in lyo_all if f["balanced_accuracy"] is not None])
    mean_sev = _mean_std([f["balanced_accuracy"] for f in lyo_sev if f["balanced_accuracy"] is not None])
    mean_ams = _mean_std([f["balanced_accuracy"] for f in lyo_ams if f["balanced_accuracy"] is not None])

    def _fmt(pair: Optional[Tuple[float, float]]) -> str:
        return "-" if pair is None else f"{pair[0] * 100:.1f}±{pair[1] * 100:.1f}"

    add("(a) Does severity dominate across ALL years?")
    add(
        f"    leave-year-out mean balanced accuracy: all={_fmt(mean_all)}, "
        f"severity_only={_fmt(mean_sev)}, all_minus_severity={_fmt(mean_ams)} "
        f"(chance=50.0)"
    )
    per_year_sev = {}
    per_year_all = {}
    per_year_ams = {}
    for f in lyo_sev:
        per_year_sev[f["year"]] = f["balanced_accuracy"]
    for f in lyo_all:
        per_year_all[f["year"]] = f["balanced_accuracy"]
    for f in lyo_ams:
        per_year_ams[f["year"]] = f["balanced_accuracy"]
    add("    per-year balanced accuracy (severity_only / all / all_minus_severity):")
    for y in sorted(per_year_all):
        add(
            f"      {y}: {_fmt_pct(per_year_sev.get(y))} / {_fmt_pct(per_year_all.get(y))} "
            f"/ {_fmt_pct(per_year_ams.get(y))}"
        )

    add("\n(b) Does any feature add stable held-out signal beyond severity?")
    sev_minus_ams = (
        None
        if mean_sev is None or mean_ams is None
        else (mean_sev[0] - mean_ams[0], (mean_sev[1] ** 2 + mean_ams[1] ** 2) ** 0.5)
    )
    add(
        f"    leave-year-out: severity_only - all_minus_severity = "
        f"{'-' if sev_minus_ams is None else f'{sev_minus_ams[0] * 100:.1f}±{sev_minus_ams[1] * 100:.1f}'} pts "
        f"(>0 means severity adds signal beyond the other features)"
    )
    rs_all = rs["all"]["balanced_accuracy"]
    rs_sev = rs["severity_only"]["balanced_accuracy"]
    rs_ams = rs["all_minus_severity"]["balanced_accuracy"]
    add(
        f"    region-season holdout ({holdout_region}/{holdout_season}): "
        f"all={_fmt_pct(rs_all)}, severity_only={_fmt_pct(rs_sev)}, "
        f"all_minus_severity={_fmt_pct(rs_ams)}"
    )
    str_all = strata["all"]
    str_sev = strata["severity_only"]
    add(
        f"    severity strata (all vs severity_only, per tercile): "
        f"low {_fmt_pct(str_all['low']['balanced_accuracy'])}/{_fmt_pct(str_sev['low']['balanced_accuracy'])}, "
        f"mid {_fmt_pct(str_all['mid']['balanced_accuracy'])}/{_fmt_pct(str_sev['mid']['balanced_accuracy'])}, "
        f"high {_fmt_pct(str_all['high']['balanced_accuracy'])}/{_fmt_pct(str_sev['high']['balanced_accuracy'])}"
    )
    matched_all = matched["per_ablation"]["all"]["matched_balanced_accuracy"]
    matched_full = matched["per_ablation"]["all"]["full_balanced_accuracy"]
    add(
        f"    severity-matched subset: all-model balanced accuracy "
        f"{_fmt_ms(matched_all)} vs {_fmt_ms(matched_full)} on the full fold"
    )

    add("\n(c) Honest balanced-accuracy ceiling on this multi-year benchmark:")
    best_name = max(ABLATION_ORDER, key=lambda n: (_mean_std([f["balanced_accuracy"] for f in lyo[n] if f["balanced_accuracy"] is not None]) or (0.0, 0.0))[0])
    best_ms = _mean_std([f["balanced_accuracy"] for f in lyo[best_name] if f["balanced_accuracy"] is not None])
    add(f"    best leave-year-out ablation: {best_name} at {_fmt_ms(best_ms)} (chance = 50.0)")
    ab = result["abstention"]
    target = 0.80
    tau_at_target = None
    for row in ab.values():
        if row["balanced_accuracy"] is not None and row["balanced_accuracy"] >= target:
            tau_at_target = row
    if tau_at_target is not None:
        add(
            f"    abstention reaches {target * 100:.0f}% balanced accuracy at tau={tau_at_target['tau']:.1f} "
            f"(coverage {tau_at_target['coverage'] * 100:.1f}%)"
        )
    else:
        best_ab = max(ab.values(), key=lambda r: r["balanced_accuracy"] or 0.0)
        add(
            f"    abstention NEVER reaches {target * 100:.0f}% balanced accuracy; best is "
            f"{best_ab['balanced_accuracy'] * 100:.1f}% at tau={best_ab['tau']:.1f} "
            f"(coverage {best_ab['coverage'] * 100:.1f}%)"
        )

    add("\n(d) Abstention curve (can we get 80% balanced accuracy at 50% coverage?):")
    # Coverage decreases with tau; interpolate balanced accuracy at coverage=0.5
    # between the two adjacent tau points.
    points = [
        (row["coverage"] or 0.0, row["balanced_accuracy"] or 0.0)
        for row in ab.values()
    ]
    cov_50 = None
    for i in range(1, len(points)):
        c_prev, b_prev = points[i - 1]
        c_cur, b_cur = points[i]
        if c_prev >= 0.50 >= c_cur:
            frac = (c_prev - 0.50) / (c_prev - c_cur) if c_prev != c_cur else 0.0
            cov_50 = b_prev + frac * (b_cur - b_prev)
            break
    if cov_50 is None and points:
        cov_50 = points[-1][1]
    add(
        f"    interpolated balanced accuracy at 50% coverage: "
        f"{'-' if cov_50 is None else f'{cov_50 * 100:.1f}%'}"
    )

    # (e) 850 hPa transport-wind signal: does t850_only beat chance, and does
    # adding transport wind to `all` help on every held-out regime?
    lyo_t850 = lyo["t850_only"]
    lyo_all_t850 = lyo["all_plus_t850"]
    mean_t850 = _mean_std([f["balanced_accuracy"] for f in lyo_t850 if f["balanced_accuracy"] is not None])
    mean_all_t850 = _mean_std([f["balanced_accuracy"] for f in lyo_all_t850 if f["balanced_accuracy"] is not None])
    add("\n(e) Does 850 hPa transport wind carry smoke-vs-dust source signal?")
    add(
        f"    t850_only leave-year-out mean: {_fmt(mean_t850)} (chance=50.0) -> "
        f"{'beats chance' if mean_t850 is not None and mean_t850[0] > 0.5 else 'at/below chance'}"
    )
    delta = (
        None
        if mean_all is None or mean_all_t850 is None
        else (mean_all_t850[0] - mean_all[0], (mean_all_t850[1] ** 2 + mean_all[1] ** 2) ** 0.5)
    )
    add(
        f"    all_plus_t850 - all (leave-year-out mean): "
        f"{'-' if delta is None else f'{delta[0] * 100:+.1f}±{delta[1] * 100:.1f}'} pts "
        f"(>0 means transport wind adds held-out signal beyond `all`)"
    )
    rs_all_t850 = rs["all_plus_t850"]["balanced_accuracy"]
    add(
        f"    region-season holdout ({holdout_region}/{holdout_season}): "
        f"all={_fmt_pct(rs_all)}, all_plus_t850={_fmt_pct(rs_all_t850)}"
    )
    matched_all_t850 = matched["per_ablation"]["all_plus_t850"]["matched_balanced_accuracy"]
    add(
        f"    severity-matched subset: all_plus_t850 {_fmt_ms(matched_all_t850)} "
        f"vs all {_fmt_ms(matched_all)} (matched bal_acc)"
    )
    per_year_t850 = {f["year"]: f["balanced_accuracy"] for f in lyo_t850}
    per_year_all_t850 = {f["year"]: f["balanced_accuracy"] for f in lyo_all_t850}
    add("    per-year balanced accuracy (t850_only / all_plus_t850):")
    for y in sorted(per_year_all):
        add(
            f"      {y}: {_fmt_pct(per_year_t850.get(y))} / {_fmt_pct(per_year_all_t850.get(y))}"
        )

    # (f) The Track-B dust bet: does "gusty + dry" separate dust from smoke on
    # the gates where severity is controlled for?
    lyo_dust = lyo["dust_opportunity"]
    mean_dust = _mean_std([f["balanced_accuracy"] for f in lyo_dust if f["balanced_accuracy"] is not None])
    lyo_all_dust = lyo["all_plus_dust"]
    mean_all_dust = _mean_std([f["balanced_accuracy"] for f in lyo_all_dust if f["balanced_accuracy"] is not None])
    add("\n(f) Does the dust-opportunity climatology (gust + 30-day precip) carry")
    add("    smoke-vs-dust signal where severity is controlled for?")
    add(
        f"    dust_opportunity leave-year-out mean: {_fmt(mean_dust)} (chance=50.0) -> "
        f"{'beats chance' if mean_dust is not None and mean_dust[0] > 0.5 else 'at/below chance'}"
    )
    delta_dust = (
        None
        if mean_all is None or mean_all_dust is None
        else (mean_all_dust[0] - mean_all[0], (mean_all_dust[1] ** 2 + mean_all[1] ** 2) ** 0.5)
    )
    add(
        f"    all_plus_dust - all (leave-year-out mean): "
        f"{'-' if delta_dust is None else f'{delta_dust[0] * 100:+.1f}±{delta_dust[1] * 100:.1f}'} pts "
        f"(>0 means gust/precip add held-out signal beyond `all`)"
    )
    rs_dust = rs["dust_opportunity"]["balanced_accuracy"]
    rs_all_dust = rs["all_plus_dust"]["balanced_accuracy"]
    add(
        f"    region-season holdout ({holdout_region}/{holdout_season}): "
        f"severity_only={_fmt_pct(rs_sev)}, dust_opportunity={_fmt_pct(rs_dust)}, "
        f"all_plus_dust={_fmt_pct(rs_all_dust)}"
    )
    matched_dust = matched["per_ablation"]["dust_opportunity"]["matched_balanced_accuracy"]
    matched_sev = matched["per_ablation"]["severity_only"]["matched_balanced_accuracy"]
    matched_all_dust = matched["per_ablation"]["all_plus_dust"]["matched_balanced_accuracy"]
    add(
        f"    severity-matched subset: dust_opportunity {_fmt_ms(matched_dust)} vs "
        f"severity_only {_fmt_ms(matched_sev)}; all_plus_dust {_fmt_ms(matched_all_dust)}"
    )
    per_year_dust = {f["year"]: f["balanced_accuracy"] for f in lyo_dust}
    add("    per-year balanced accuracy (dust_opportunity):")
    for y in sorted(per_year_all):
        add(f"      {y}: {_fmt_pct(per_year_dust.get(y))}")

    return "\n".join(lines)


def specbench_command(db_path: str) -> str:
    """Open the store, run the benchmark, and return the formatted report."""
    def _progress(i: int, total: int) -> None:
        if i % 20000 == 0 or i == total:
            print(f"  assembled {i}/{total} joinable site-days", file=sys.stderr)

    with AccuracyStore(db_path) as store:
        result = run_specbench(store, progress=_progress)
    return format_report(result)
