"""
Centralized attribution threshold parameters for the Upwind engine.

All numeric constants used in scoring/classification comparisons live here so
they can be audited and tuned in one place. This module is a LEAF: it imports
only from the stdlib, never from the rest of the application, so it cannot
create import cycles.
"""

# ---------------------------------------------------------------------------
# OpenAQ monitor concentration thresholds (engine/score.py)
# Applied only when monitor status is present.
# ---------------------------------------------------------------------------

# PM2.5/PM10 ratio below this indicates coarse-dominated dust
OPENAQ_DUST_RATIO_MAX = 0.35
# PM2.5/PM10 ratio at/above this indicates fine-dominated smoke
OPENAQ_SMOKE_RATIO_MIN = 0.70
# Traffic combustion tracer
OPENAQ_NO2_PPB = 50.0
# Industrial point-source tracer
OPENAQ_SO2_PPB = 75.0
# Combustion tracer
OPENAQ_CO_PPM = 2.0
# Ground-level ozone threshold
OPENAQ_O3_PPB = 70.0
# Same-hour percentile at/above which a reading is anomalous
OPENAQ_SAME_HOUR_PERCENTILE = 90.0
# Multiple of the same-hour median a reading must reach to count as anomalous
OPENAQ_SAME_HOUR_FACTOR = 2.0
# Suppress monitor scoring boosts when reading conflicts with reported AQI band
OPENAQ_MEASURED_CONFLICT_FACTOR = 0.5

# ---------------------------------------------------------------------------
# OpenAQ service query parameters (services/openaq.py)
# ---------------------------------------------------------------------------

# Preferred reference-monitor search radius, in meters
OPENAQ_PREFERRED_RADIUS_M = 10_000
# Widened reference-monitor search radius, in meters
OPENAQ_RADIUS_M = 25_000
# Hourly monitor readings older than this are dropped (3h)
MAX_READING_AGE_S = 3 * 3600
# Baseline historical window (days) for daily-percentile context
BASELINE_DAYS = 365
# Same-hour-of-day baseline window (days)
SAME_HOUR_WINDOW_DAYS = 30
# Minimum same-hour samples required to avoid noisy percentile votes
SAME_HOUR_MIN_SAMPLES = 5
# Completeness threshold (%) for aggregated records
MIN_PERCENT_COMPLETE = 75.0

# ---------------------------------------------------------------------------
# WFIGS incident registry (services/wfigs.py)
# ---------------------------------------------------------------------------

# Maximum distance (mi) for a WFIGS incident to count as a smoke source
WFIGS_MAX_RADIUS_MILES = 300.0
# Incidents above this containment (%) are skipped as smoke sources
WFIGS_MAX_CONTAINMENT_PCT = 90.0

# ---------------------------------------------------------------------------
# Fire-source relevance weighting (services/wfigs.py, services/firms.py)
# ---------------------------------------------------------------------------

# Multiplier applied to upwind-aligned WFIGS incidents
WFIGS_UPWIND_BONUS = 4.0
# Fallback size (acres) when a WFIGS incident has no IncidentSize
WFIGS_DEFAULT_SIZE_ACRES = 50.0
# Minimum activity (1 - containment) weight for WFIGS incidents
WFIGS_ACTIVITY_FLOOR = 0.1
# Added to distance (mi) to avoid div-by-zero in the WFIGS relevance decay
WFIGS_RELEVANCE_EPS_MILES = 1.0
# Multiplier applied to upwind-aligned FIRMS hotspots
FIRMS_UPWIND_BONUS = 4.0
# Added to distance (mi) to avoid div-by-zero in the FIRMS relevance decay
FIRMS_RELEVANCE_EPS_MILES = 1.0

# ---------------------------------------------------------------------------
# FIRMS hotspot feed (services/firms.py)
# ---------------------------------------------------------------------------

# Hotspots within +/-90 deg of the upwind bearing count as upwind
UPWIND_SECTOR_WIDTH_DEG = 90.0
# Floor search radius (mi) so calm conditions still cover nearby fires
FIRMS_MIN_RADIUS_MILES = 75.0
# Ceiling search radius (mi)
FIRMS_MAX_RADIUS_MILES = 150.0
# Search radius multiplier: wind_speed_mph * factor = radius (mi)
FIRMS_RADIUS_WIND_FACTOR = 5.0
# Default wind speed (mph) used when the wind reading is missing
FIRMS_DEFAULT_WIND_MPH = 10.0

# ---------------------------------------------------------------------------
# FIRMS recency / confidence / clustering (services/firms.py)
# ---------------------------------------------------------------------------

# Window length; hotspots older than this are dropped
FIRMS_MAX_AGE_HOURS = 48.0
# Exponential age decay half-life
FIRMS_RECENCY_HALF_LIFE_HOURS = 12.0
# Minimum recency weight
FIRMS_RECENCY_FLOOR = 0.1
# Confidence weight: low -> dropped (weight 0); unknown labels fall back to 1.0
FIRMS_CONFIDENCE_WEIGHT = {"low": 0.0, "nominal": 0.7, "high": 1.0}
# Pixels within this distance merge into one cluster
FIRMS_CLUSTER_RADIUS_KM = 2.0
# Clusters below summed FRP are ignored
FIRMS_MIN_CLUSTER_FRP = 1.0
# Persistence multiplier growth per extra detection (beyond the first)
FIRMS_PERSISTENCE_STEP = 0.2
# Persistence multiplier ceiling (bounded so overpass count can't dominate intensity)
FIRMS_PERSISTENCE_CAP = 1.6

# ---------------------------------------------------------------------------
# WFIGS corroboration radius used by scoring (engine/score.py)
# ---------------------------------------------------------------------------

# Maximum distance (mi) for non-aligned WFIGS fire transport corroboration
WFIGS_CORROBORATION_RADIUS_MILES = 150.0

# ---------------------------------------------------------------------------
# EPA AQI breakpoints (engine/score.py)
# FIXED — EPA regulatory values from the AQS breakpoint table; NOT tunable.
# ---------------------------------------------------------------------------

# Surface PM AQI at/above which conditions are Unhealthy or worse
EXTREME_PM_AQI = 150
# AQI lower-bound PM2.5 concentrations (µg/m³) by AQI band
PM25_AQI_LOWER_BOUNDS = (
    (51, 9.1), (101, 35.5), (151, 55.5), (201, 125.5), (301, 225.5), (401, 325.5)
)
# AQI lower-bound PM10 concentrations (µg/m³) by AQI band
PM10_AQI_LOWER_BOUNDS = (
    (51, 55.0), (101, 155.0), (151, 255.0), (201, 355.0), (301, 425.0)
)

# ---------------------------------------------------------------------------
# Aerosol Optical Depth (AOD) thresholds (engine/score.py, engine/signals.py)
# ---------------------------------------------------------------------------

# AOD at/above this is at least light haze / a present column plume
AOD_HAZE = 0.2
# AOD at/above this is a medium-density plume
AOD_MEDIUM = 0.4
# AOD at/above this is a heavy plume
AOD_HEAVY = 0.8

# ---------------------------------------------------------------------------
# Weather thresholds (engine/score.py, engine/signals.py)
# ---------------------------------------------------------------------------

# Wind speed below this (mph) counts as calm
CALM_WIND_SPEED_MPH = 4.0
# Wind speed at/above this (mph) is high enough to loft dust
HIGH_WIND_SPEED_MPH = 15.0
# Wind speed at/above this (mph) supports medium-confidence dust
DUST_WIND_SPEED_MPH = 8.0
# Temperature below this (°F) counts as cold (winter stagnation)
COLD_TEMP_F = 55.0
# Temperature below this (°F) suppresses photochemical ozone generation
OZONE_COOL_TEMP_F = 75.0
# Temperature at/above this (°F) counts as a hot day for ozone formation
OZONE_HOT_TEMP_F = 85.0
# Boundary layer height below this (m) indicates trapped near-surface air
SHALLOW_BOUNDARY_LAYER_M = 500

# ---------------------------------------------------------------------------
# AQI / percentile thresholds (engine/score.py, engine/signals.py)
# ---------------------------------------------------------------------------

# AQI above this counts as elevated PM (and at/at-below this is "Good")
AQI_ELEVATED = 50
# Daily percentile at/above which a reading is well above typical for the area
DAILY_PERCENTILE_HIGH = 90.0
# Monitor distance at/below which (km) a low reading is a disclosed mismatch
CONFLICT_DISTANCE_KM = 10.0

# ---------------------------------------------------------------------------
# Fire evidence signal counts (engine/score.py)
# ---------------------------------------------------------------------------

# At least this many verified fire signals yields a medium smoke score
FIRE_SIGNAL_MIN_COUNT = 1
# At least this many verified fire signals (upwind FIRMS) yields high smoke
FIRE_SIGNAL_HIGH_COUNT = 2
