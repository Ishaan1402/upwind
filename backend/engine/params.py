"""
Centralized attribution threshold parameters for the Upwind engine.

All numeric constants used in scoring/classification comparisons live here so
they can be audited and tuned in one place. This module is a LEAF: it imports
only from the stdlib, never from the rest of the application, so it cannot
create import cycles.

The thresholds are carried by a frozen :class:`Params` dataclass. Scoring
consumers read the active params via :func:`get_params` (the default is
``DEFAULT``, overridable with :func:`use_params` for tuning runs); ground-truth
label derivation always uses the frozen ``LABEL_PARAMS`` so tuning the scorer
never moves the labels.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping, Tuple


@dataclass(frozen=True)
class Params:
    # ---------------------------------------------------------------------
    # OpenAQ monitor concentration thresholds (engine/score.py)
    # Applied only when monitor status is present.
    # ---------------------------------------------------------------------

    # PM2.5/PM10 ratio below this indicates coarse-dominated dust
    openaq_dust_ratio_max: float = 0.35
    # PM2.5/PM10 ratio at/above this indicates fine-dominated smoke
    openaq_smoke_ratio_min: float = 0.70
    # Traffic combustion tracer
    openaq_no2_ppb: float = 50.0
    # Industrial point-source tracer
    openaq_so2_ppb: float = 75.0
    # Combustion tracer
    openaq_co_ppm: float = 2.0
    # Ground-level ozone threshold
    openaq_o3_ppb: float = 70.0
    # Same-hour percentile at/above which a reading is anomalous
    openaq_same_hour_percentile: float = 90.0
    # Multiple of the same-hour median a reading must reach to count as anomalous
    openaq_same_hour_factor: float = 2.0
    # Suppress monitor scoring boosts when reading conflicts with reported AQI band
    openaq_measured_conflict_factor: float = 0.5

    # ---------------------------------------------------------------------
    # OpenAQ service query parameters (services/openaq.py)
    # ---------------------------------------------------------------------

    # Preferred reference-monitor search radius, in meters
    openaq_preferred_radius_m: int = 10_000
    # Widened reference-monitor search radius, in meters
    openaq_radius_m: int = 25_000
    # Hourly monitor readings older than this are dropped (3h)
    max_reading_age_s: int = 3 * 3600
    # Baseline historical window (days) for daily-percentile context
    baseline_days: int = 365
    # Same-hour-of-day baseline window (days)
    same_hour_window_days: int = 30
    # Minimum same-hour samples required to avoid noisy percentile votes
    same_hour_min_samples: int = 5
    # Completeness threshold (%) for aggregated records
    min_percent_complete: float = 75.0

    # ---------------------------------------------------------------------
    # WFIGS incident registry (services/wfigs.py)
    # ---------------------------------------------------------------------

    # Maximum distance (mi) for a WFIGS incident to count as a smoke source
    wfigs_max_radius_miles: float = 300.0
    # Incidents above this containment (%) are skipped as smoke sources
    wfigs_max_containment_pct: float = 90.0

    # ---------------------------------------------------------------------
    # Fire-source relevance weighting (services/wfigs.py, services/firms.py)
    # ---------------------------------------------------------------------

    # Multiplier applied to upwind-aligned WFIGS incidents
    wfigs_upwind_bonus: float = 4.0
    # Fallback size (acres) when a WFIGS incident has no IncidentSize
    wfigs_default_size_acres: float = 50.0
    # Minimum activity (1 - containment) weight for WFIGS incidents
    wfigs_activity_floor: float = 0.1
    # Added to distance (mi) to avoid div-by-zero in the WFIGS relevance decay
    wfigs_relevance_eps_miles: float = 1.0
    # Multiplier applied to upwind-aligned FIRMS hotspots
    firms_upwind_bonus: float = 4.0
    # Added to distance (mi) to avoid div-by-zero in the FIRMS relevance decay
    firms_relevance_eps_miles: float = 1.0

    # ---------------------------------------------------------------------
    # FIRMS hotspot feed (services/firms.py)
    # ---------------------------------------------------------------------

    # Hotspots within +/-90 deg of the upwind bearing count as upwind
    upwind_sector_width_deg: float = 90.0
    # Floor search radius (mi) so calm conditions still cover nearby fires
    firms_min_radius_miles: float = 75.0
    # Ceiling search radius (mi)
    firms_max_radius_miles: float = 150.0
    # Search radius multiplier: wind_speed_mph * factor = radius (mi)
    firms_radius_wind_factor: float = 5.0
    # Default wind speed (mph) used when the wind reading is missing
    firms_default_wind_mph: float = 10.0

    # ---------------------------------------------------------------------
    # FIRMS recency / confidence / clustering (services/firms.py)
    # ---------------------------------------------------------------------

    # Window length; hotspots older than this are dropped
    firms_max_age_hours: float = 48.0
    # Exponential age decay half-life
    firms_recency_half_life_hours: float = 12.0
    # Minimum recency weight
    firms_recency_floor: float = 0.1
    # Confidence weight: low -> dropped (weight 0); unknown labels fall back to 1.0
    # Immutable (MappingProxyType) so callers can't mutate global scoring in place.
    firms_confidence_weight: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({"low": 0.0, "nominal": 0.7, "high": 1.0})
    )
    # Pixels within this distance merge into one cluster
    firms_cluster_radius_km: float = 2.0
    # Clusters below summed FRP are ignored
    firms_min_cluster_frp: float = 1.0
    # Persistence multiplier growth per extra detection (beyond the first)
    firms_persistence_step: float = 0.2
    # Persistence multiplier ceiling (bounded so overpass count can't dominate intensity)
    firms_persistence_cap: float = 1.6

    # ---------------------------------------------------------------------
    # WFIGS corroboration radius used by scoring (engine/score.py)
    # ---------------------------------------------------------------------

    # Maximum distance (mi) for non-aligned WFIGS fire transport corroboration
    wfigs_corroboration_radius_miles: float = 150.0

    # ---------------------------------------------------------------------
    # EPA AQI breakpoints (engine/score.py)
    # FIXED — EPA regulatory values from the AQS breakpoint table; NOT tunable.
    # ---------------------------------------------------------------------

    # Surface PM AQI at/above which conditions are Unhealthy or worse
    extreme_pm_aqi: int = 150
    # AQI lower-bound PM2.5 concentrations (µg/m³) by AQI band
    pm25_aqi_lower_bounds: Tuple[Tuple[int, float], ...] = (
        (51, 9.1), (101, 35.5), (151, 55.5), (201, 125.5), (301, 225.5), (401, 325.5)
    )
    # AQI lower-bound PM10 concentrations (µg/m³) by AQI band
    pm10_aqi_lower_bounds: Tuple[Tuple[int, float], ...] = (
        (51, 55.0), (101, 155.0), (151, 255.0), (201, 355.0), (301, 425.0)
    )

    # ---------------------------------------------------------------------
    # Aerosol Optical Depth (AOD) thresholds (engine/score.py, engine/signals.py)
    # ---------------------------------------------------------------------

    # AOD at/above this is at least light haze / a present column plume
    aod_haze: float = 0.2
    # AOD at/above this is a medium-density plume
    aod_medium: float = 0.4
    # AOD at/above this is a heavy plume
    aod_heavy: float = 0.8

    # ---------------------------------------------------------------------
    # Weather thresholds (engine/score.py, engine/signals.py)
    # ---------------------------------------------------------------------

    # Wind speed below this (mph) counts as calm
    calm_wind_speed_mph: float = 4.0
    # Wind speed at/above this (mph) is high enough to loft dust
    high_wind_speed_mph: float = 15.0
    # Wind speed at/above this (mph) supports medium-confidence dust
    dust_wind_speed_mph: float = 8.0
    # Temperature below this (°F) counts as cold (winter stagnation)
    cold_temp_f: float = 55.0
    # Temperature below this (°F) suppresses photochemical ozone generation
    ozone_cool_temp_f: float = 75.0
    # Temperature at/above this (°F) counts as a hot day for ozone formation
    ozone_hot_temp_f: float = 85.0
    # Boundary layer height below this (m) indicates trapped near-surface air
    shallow_boundary_layer_m: int = 500

    # ---------------------------------------------------------------------
    # AQI / percentile thresholds (engine/score.py, engine/signals.py)
    # FIXED — the AQI elevated band ("Good") follows the EPA breakpoint table
    # alongside extreme_pm_aqi and the breakpoint tuples above.
    # ---------------------------------------------------------------------

    # AQI above this counts as elevated PM (and at/at-below this is "Good")
    aqi_elevated: int = 50
    # Daily percentile at/above which a reading is well above typical for the area
    daily_percentile_high: float = 90.0
    # Monitor distance at/below which (km) a low reading is a disclosed mismatch
    conflict_distance_km: float = 10.0

    # ---------------------------------------------------------------------
    # Fire evidence signal counts (engine/score.py)
    # ---------------------------------------------------------------------

    # At least this many verified fire signals yields a medium smoke score
    fire_signal_min_count: int = 1
    # At least this many verified fire signals (upwind FIRMS) yields high smoke
    fire_signal_high_count: int = 2


# Default scoring/tuning params.
DEFAULT: Params = Params()

# Phase 2 tuning (2020 Western US holdout, see PLAN.md): lowering the dust
# loft wind 15 -> 12 mph improves dust recall (F1 .47 -> .55 on the 20% site
# holdout, .40 -> .50 on the disjoint 80% set) with no per-class regression.
# LABEL_PARAMS keeps the original 15.0 (frozen).
DEFAULT = replace(DEFAULT, high_wind_speed_mph=12.0)

# Frozen params used ONLY by ground-truth label derivation so tuning the scorer
# never moves the labels. Distinct instance (same values) as DEFAULT.
LABEL_PARAMS: Params = Params()

_active: ContextVar[Params] = ContextVar("params", default=DEFAULT)


def get_params() -> Params:
    return _active.get()


@contextmanager
def use_params(params: Params):
    token = _active.set(params)
    try:
        yield
    finally:
        _active.reset(token)
