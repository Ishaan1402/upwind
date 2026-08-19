"""Canonical record types for the accuracy-evaluation pipeline.

Records carry raw values exactly as published by the upstream source; unit
conversion / concentration interpretation is deferred to label derivation.
Keep this module dependency-free (stdlib only) so it stays importable from
any tooling.
"""

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class AqsDailyRecord:
    """One row of an EPA AQS daily summary file (daily_<param>_YYYY.csv).

    Values mirror the published CSV columns faithfully: concentrations and
    units are stored unnormalized, and empty/negative AQI values are stored as
    None. ``site_id`` is the canonical zero-padded ``SS-CCC-NNNN`` form built
    from State Code / County Code / Site Num.
    """

    site_id: str
    state_code: str
    county_code: str
    site_num: str
    parameter_code: str
    parameter_name: str
    poc: Optional[int]
    lat: Optional[float]
    lon: Optional[float]
    date_local: str  # "YYYY-MM-DD"
    concentration: Optional[float]
    units: Optional[str]
    aqi: Optional[int]
    method_code: Optional[str]

    @property
    def natural_key(self) -> Tuple[str, str, str, Optional[int]]:
        """Natural key used for idempotent SQLite upserts."""
        return (self.site_id, self.parameter_code, self.date_local, self.poc)


@dataclass(frozen=True)
class WeatherDailyRecord:
    """One day of archived weather at a site (Open-Meteo ERA5 archive).

    Values mirror the requested daily aggregates in the units asked for
    (Fahrenheit / mph / mm): ``tmax_f``/``tmin_f`` are 2m temperature extremes,
    ``wind_max_mph``/``wind_dir_dominant_deg`` the 10m max wind speed and
    dominant wind direction, ``precipitation_mm`` the daily precipitation sum
    and ``wind_gust_max_mph`` the 10m max wind gust. Days with no value for a
    variable carry None.
    """

    site_id: str
    lat: float
    lon: float
    date_local: str  # "YYYY-MM-DD"
    tmax_f: Optional[float]
    tmin_f: Optional[float]
    wind_max_mph: Optional[float]
    wind_dir_dominant_deg: Optional[int]
    precipitation_mm: Optional[float] = None
    wind_gust_max_mph: Optional[float] = None

    @property
    def natural_key(self) -> Tuple[str, str]:
        """Natural key used for idempotent SQLite upserts."""
        return (self.site_id, self.date_local)


@dataclass(frozen=True)
class TransportWindRecord:
    """One day of 850 hPa transport-layer wind at a site (NCEP/NCAR
    Reanalysis-1 daily u/v components, 2.5° grid, via NOAA PSL THREDDS NCSS).

    ``u850``/``v850`` are the eastward/northward wind components in m/s at
    850 hPa; the derived speed / meteorological direction are computed
    downstream (specbench features), so the store keeps the raw vector.
    ``source`` records provenance (``"ncep_daily"`` for the daily reanalysis
    averages; ``"ncep_monthly"`` for the monthly-mean fallback, when used).
    Days with a missing component carry None for that field.
    """

    site_id: str
    date_local: str  # "YYYY-MM-DD"
    u850: Optional[float]
    v850: Optional[float]
    source: str

    @property
    def natural_key(self) -> Tuple[str, str]:
        """Natural key used for idempotent SQLite upserts."""
        return (self.site_id, self.date_local)


@dataclass(frozen=True)
class FirmsHotspotRecord:
    """One thermal-anomaly detection as published by NASA FIRMS.

    ``acq_datetime`` is the acquisition time as an ISO8601 UTC string (e.g.
    ``2026-07-27T09:40:00+00:00``) derived from the CSV ``acq_date``/``acq_time``
    fields. lat/lon are stored at full float precision so distinct detections
    that differ only past the 4th decimal are not silently merged.
    """

    lat: float
    lon: float
    frp: float
    acq_datetime: str  # ISO8601 UTC
    confidence: Optional[str]
    satellite: Optional[str]
    daynight: Optional[str]

    @property
    def natural_key(self) -> Tuple[float, float, str, Optional[str]]:
        """Natural key used for idempotent SQLite upserts."""
        return (self.lat, self.lon, self.acq_datetime, self.satellite)


@dataclass(frozen=True)
class SpeciationRow:
    """One row of an EPA AQS speciation daily file (daily_SPEC_YYYY.csv).

    Speciation is the PM2.5 chemical-composition dataset (elements, ions,
    carbon) used by IMPROVE-type analyses. ``site_id`` is the canonical
    zero-padded ``SS-CCC-NNNN`` form built from State Code / County Code /
    Site Num, exactly as ``AqsDailyRecord``. The natural key drops POC/method
    because label derivation keys on one concentration per parameter per day
    (INSERT OR REPLACE keeps re-ingestion idempotent).

    ``lat``/``lon`` mirror the published Latitude/Longitude columns (inside the
    ingest bbox) and feed the ``speciation_sites`` table used to fetch weather
    for IMPROVE sites; they are not stored on the ``speciation`` row itself.
    """

    site_id: str
    date_local: str  # "YYYY-MM-DD"
    parameter_code: str
    parameter_name: Optional[str]
    method_code: Optional[str]
    method_name: Optional[str]
    concentration: Optional[float]
    units: Optional[str]
    lat: Optional[float] = None
    lon: Optional[float] = None

    @property
    def natural_key(self) -> Tuple[str, str, str]:
        """Natural key used for idempotent SQLite upserts."""
        return (self.site_id, self.date_local, self.parameter_code)


@dataclass(frozen=True)
class HmsSmokeRecord:
    """One analyst-drawn HMS smoke-plume polygon for a day.

    ``density`` is the canonical label (``"light"`` | ``"medium"`` |
    ``"heavy"``) and ``geometry_json`` a GeoJSON Polygon geometry object
    serialized as text, e.g. ``{"type":"Polygon","coordinates":[[[lon,lat],...]]}``
    with the exterior ring first, followed by any holes.
    """

    date_local: str  # "YYYY-MM-DD"
    density: str  # "light" | "medium" | "heavy"
    geometry_json: str  # GeoJSON Polygon geometry object

    @property
    def natural_key(self) -> Tuple[str, str, str]:
        """Natural key used for idempotent SQLite upserts."""
        geometry_hash = hashlib.sha1(self.geometry_json.encode()).hexdigest()[:16]
        return (self.date_local, self.density, geometry_hash)


@dataclass(frozen=True)
class Observation:
    """A single day's aggregated air-quality state at a site.

    Built from the day's ``AqsDailyRecord`` rows by ``build_observation``:
    ``aqi`` is the max non-null AQI across parameters (None when no row carries
    an AQI) and ``primary_pollutant`` the pollutant carrying that max (ties
    broken by canonical order; empty when there is no AQI). The per-pollutant
    dicts keep the raw daily AQI and concentration for each parameter so
    reasoning/audit code can inspect the contribution of each pollutant.
    """

    aqi: Optional[int]
    primary_pollutant: str
    pollutant_aqi: Dict[str, Optional[int]]  # pollutant name -> daily AQI
    concentrations: Dict[str, Optional[float]]  # pollutant name -> daily concentration


@dataclass(frozen=True)
class LabelRecord:
    """A derived ground-truth label for one site-day.

    Labels are DERIVED from archived evidence (AQS daily summaries, weather,
    HMS smoke, FIRMS upwind fires) rather than hand-curated; ``reasoning`` is a
    one-line description of the single rule that fired. ``precision_tier`` is
    ``"validated"`` while labels are AQS-backed. ``label_kind`` marks that the
    label is ``"rule_derived"`` — produced by re-applying production thresholds
    to the same archives the scorer consumes, so it measures self-consistency,
    not independently-verified accuracy. ``aqi`` is None when the day had no
    determinable AQI.
    """

    site_id: str
    date_local: str  # "YYYY-MM-DD"
    aqi: Optional[int]
    primary_pollutant: str
    label: str
    precision_tier: str
    reasoning: str
    label_kind: str = "rule_derived"

    @property
    def natural_key(self) -> Tuple[str, str]:
        """Natural key used for idempotent SQLite upserts."""
        return (self.site_id, self.date_local)


@dataclass(frozen=True)
class PredictionRecord:
    """One stored scorer-vs-label evaluation outcome for a site-day.

    ``true_label`` is the derived ground-truth label (``labels.LABEL_CLASSES``),
    ``predicted_label`` the top-ranked scorer hypothesis id, and ``top_score`` /
    ``top_confidence`` the top hypothesis's numeric score and confidence band.
    """

    site_id: str
    date_local: str  # "YYYY-MM-DD"
    true_label: str
    predicted_label: str
    top_score: Optional[float]
    top_confidence: Optional[str]

    @property
    def natural_key(self) -> Tuple[str, str]:
        """Natural key used for idempotent SQLite upserts."""
        return (self.site_id, self.date_local)
