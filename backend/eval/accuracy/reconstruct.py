"""Reconstruct production scorer inputs from stored archive records.

This module replays the evidence pipeline *offline*: for a stored site-day it
loads the canonical records (AQS daily summaries, archived weather, FIRMS
hotspots, HMS smoke polygons) and rebuilds the exact dict shapes the
production scorer consumes — ``build_evidence_signals`` (the same assembler
used by ``/api/why``) followed by ``score_hypotheses``.

Feed adapters that have no historical counterpart yet (AOD, WFIGS, Census
place context, news/web search) are represented by their "unavailable" shapes
so scoring sees a feed outage rather than a verified absence — mirroring how
``backend.engine.signals`` treats a raised/empty feed.

Reused rather than reimplemented:
- ``build_evidence_signals`` / ``score_hypotheses`` (the production DAG + scorer)
- ``cluster_firms_hotspots`` and the distance/bearing helpers from
  ``backend.services.firms`` (FIRMS clustering/weighting stays identical)
- ``_point_in_ring_set`` from ``backend.services.hms`` (GeoJSON ring semantics)
- ``firms_search_radius_miles`` / ``FIRMS_*`` params from ``backend.engine.params``
- ``build_observation`` from ``labels`` (daily AQI aggregation)

Only the archive-to-dict glue is implemented here.
"""

import json
import math
from datetime import date, datetime, time, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from backend.config import get_aqi_category
from backend.engine.params import (
    FIRMS_CONFIDENCE_WEIGHT,
    FIRMS_MAX_AGE_HOURS,
    UPWIND_SECTOR_WIDTH_DEG,
)
from backend.engine.score import score_hypotheses
from backend.engine.signals import build_evidence_signals
from backend.eval.accuracy.labels import build_observation
from backend.eval.accuracy.records import (
    AqsDailyRecord,
    Observation,
    WeatherDailyRecord,
)
from backend.eval.accuracy.store import AccuracyStore
from backend.services.firms import (
    angular_difference,
    bearing_degrees_to_compass,
    calculate_bearing_degrees,
    calculate_haversine_distance,
    cluster_firms_hotspots,
    firms_search_radius_miles,
)
from backend.services.hms import _point_in_ring_set

# ---------------------------------------------------------------------------
# Observation / weather dict assembly
# ---------------------------------------------------------------------------


def _observation_dict(observation: Observation) -> Dict:
    """Convert the derived ``Observation`` into the dict the engine consumes.

    ``score_hypotheses`` reads ``aqi``/``primary_pollutant``/``category`` and
    ``build_evidence_signals`` reads ``pollutants`` (name -> concentration)
    for display in the ``surface_pm_level`` signal; the scorer itself decides
    elevation from the AQI and the OpenAQ-derived signal. ``aqi`` passes
    through faithfully (None when the day had no determinable AQI) and
    ``category`` is None in that case.
    """
    return {
        "source": "EPA AQS daily summary",
        "aqi": observation.aqi,
        "primary_pollutant": observation.primary_pollutant,
        "category": (
            get_aqi_category(observation.aqi)["label"]
            if observation.aqi is not None
            else None
        ),
        "pollutants": dict(observation.concentrations),
    }


def _weather_dict(weather: Optional[WeatherDailyRecord]) -> Dict:
    """Rebuild the ``fetch_openmeteo_weather`` payload shape from the archive.

    Archived weather has no boundary-layer reading, so that key is None; the
    temperature falls back from the daily max to the daily min when the max is
    missing. None-safe: a missing weather record becomes an all-None dict (the
    engine treats it as an unavailable wind vector).
    """
    if weather is None:
        return {
            "wind_speed_mph": None,
            "wind_direction_deg": None,
            "temperature_f": None,
            "boundary_layer_height_m": None,
        }
    temperature_f = weather.tmax_f if weather.tmax_f is not None else weather.tmin_f
    return {
        "wind_speed_mph": weather.wind_max_mph,
        "wind_direction_deg": weather.wind_dir_dominant_deg,
        "temperature_f": temperature_f,
        "boundary_layer_height_m": None,
    }


# ---------------------------------------------------------------------------
# FIRMS reconstruction
# ---------------------------------------------------------------------------


def _hotspots_for_day(
    store: AccuracyStore,
    lat: float,
    lon: float,
    date_local: str,
    wind_dir_deg: Optional[float],
    radius_mi: float,
) -> List[Dict]:
    """FIRMS pixels near ``(lat, lon)`` for the day, as ``fetch_firms_hotspots``
    builds them.

    A coarse lat/lon bbox (``lat ± radius_mi/69``,
    ``lon ± radius_mi/(69*cos(lat))``) plus a NO-LOOK-AHEAD acquisition window
    of ``[date_local 23:59:59 UTC - 48h, date_local 23:59:59 UTC]`` — SQLite's
    lexical collation is correct here because ``acq_datetime`` is always
    stored as an ISO8601 UTC string. No detection from a later day may leak
    into a site-day's reconstruction. Per-pixel fields (distance/bearing/age/
    confidence weight/upwind) match the production pixel dict; per-pixel
    ``relevance`` is left 0 because clustering recomputes relevance from FRP,
    distance, recency and upwind alignment.
    """
    target_date = date.fromisoformat(date_local)

    lat_delta = radius_mi / 69.0
    lon_delta = radius_mi / (69.0 * math.cos(math.radians(lat)))

    # The site-day's end-of-day instant is the recency reference. Using the day
    # itself as the horizon (no +24h) keeps same-day detections after noon from
    # scoring as age 0 and never pulls in the next day.
    reference = datetime.combine(target_date, time(23, 59, 59), tzinfo=timezone.utc)
    start_iso = (reference - timedelta(hours=48)).isoformat()
    end_iso = reference.isoformat()

    records = store.fetch_firms_hotspots_in_bbox(
        west=lon - lon_delta,
        south=lat - lat_delta,
        east=lon + lon_delta,
        north=lat + lat_delta,
        start_iso=start_iso,
        end_iso=end_iso,
    )

    upwind_target_deg = wind_dir_deg % 360 if wind_dir_deg is not None else None

    pixels: List[Dict] = []
    for rec in records:
        acq_dt = datetime.fromisoformat(rec.acq_datetime)
        age_hours = max(0.0, (reference - acq_dt).total_seconds() / 3600.0)
        if age_hours > FIRMS_MAX_AGE_HOURS:
            continue

        # Unknown/missing confidence keeps a neutral weight of 1.0; "low"
        # detections (sun glint / false positives) are dropped outright.
        confidence_weight = FIRMS_CONFIDENCE_WEIGHT.get(rec.confidence, 1.0)
        if confidence_weight == 0.0:
            continue

        dist_km, dist_mi = calculate_haversine_distance(lat, lon, rec.lat, rec.lon)
        bearing_deg = calculate_bearing_degrees(lat, lon, rec.lat, rec.lon)
        is_upwind = (
            wind_dir_deg is None
            or angular_difference(bearing_deg, upwind_target_deg) <= UPWIND_SECTOR_WIDTH_DEG
        )

        pixels.append({
            "lat": rec.lat,
            "lon": rec.lon,
            "frp": rec.frp,
            "age_hours": round(age_hours, 1),
            "is_upwind": is_upwind,
            "confidence": rec.confidence,
            "confidence_weight": confidence_weight,
            "distance_km": round(dist_km, 1),
            "distance_miles": round(dist_mi, 1),
            "bearing": bearing_degrees_to_compass(bearing_deg),
            "bearing_deg": round(bearing_deg, 1),
            "relevance": 0.0,  # clustering recomputes relevance
        })
    return pixels


def _firms_res_from_hotspots(
    pixels: List[Dict],
    lat: float,
    lon: float,
    wind_dir_deg: Optional[float],
    radius_mi: float,
) -> Dict:
    """Shape the pixel list exactly like ``fetch_firms_hotspots`` returns.

    Clustering and the present/absent payload contract (status, hotspots capped
    at 40, clusters capped at 10, count/alignment/nearest, details string) are
    reproduced so ``build_evidence_signals``/``score_hypotheses`` consume
    byte-identical inputs. Empty or sub-threshold pixels yield a verified
    "absent" result.
    """
    if not pixels:
        return {
            "status": "absent",
            "hotspots": [],
            "count": 0,
            "total_count": 0,
            "nearest": None,
            "alignment": None,
            "details": "No active thermal hotspots detected nearby",
        }

    clusters = cluster_firms_hotspots(pixels, lat, lon)
    if not clusters:
        return {
            "status": "absent",
            "hotspots": pixels[:40],
            "count": 0,
            "total_count": 0,
            "nearest": None,
            "alignment": None,
            "details": "Detected hotspots are too weak to register (below FRP floor)",
        }

    upwind_clusters = [c for c in clusters if c["is_upwind"]]
    total_count = len(clusters)
    count = len(upwind_clusters)
    # The named source never contradicts the reported alignment: the strongest
    # upwind cluster when one exists, else the strongest overall cluster.
    nearest = upwind_clusters[0] if upwind_clusters else clusters[0]

    if upwind_clusters:
        details = (
            f"{count} upwind hotspot cluster(s) found "
            f"(strongest cluster {nearest['distance_miles']} mi {nearest['bearing']} "
            f"(FRP {nearest['frp']:.0f} MW, {nearest['detections']} detections)); "
            f"{total_count} total detected nearby"
        )
        alignment = "upwind"
    else:
        details = (
            f"{total_count} hotspot cluster(s) within ~{int(radius_mi)} mi "
            f"(strongest cluster {nearest['distance_miles']} mi {nearest['bearing']} "
            f"(FRP {nearest['frp']:.0f} MW, {nearest['detections']} detections)), "
            f"but none aligned upwind of current wind"
        )
        alignment = "nearby"

    return {
        "status": "present",
        "hotspots": pixels[:40],
        "clusters": clusters[:10],
        "count": count,
        "total_count": total_count,
        "nearest": nearest,
        "alignment": alignment,
        "details": details,
    }


# ---------------------------------------------------------------------------
# HMS / OpenAQ reconstruction
# ---------------------------------------------------------------------------


def _hms_res_from_store(
    store: AccuracyStore, date_local: str, lat: float, lon: float
) -> Dict:
    """Check the day's archived HMS smoke polygons against the site point.

    Mirrors ``check_hms_smoke_plume`` but reads polygons from the store: any
    polygon whose exterior ring contains (lon, lat) — outside every hole — marks
    the site as inside a plume; the strongest matched density (heavy > medium >
    light) wins. No polygons for the date is a feed gap, reported as
    "unavailable" rather than a verified absence.
    """
    records = store.fetch_hms_smoke(date_local=date_local)
    if not records:
        return {
            "status": "unavailable",
            "density": None,
            "details": "no HMS polygons ingested for date",
        }

    matched_densities = []
    for rec in records:
        try:
            geometry = json.loads(rec.geometry_json)
        except (TypeError, ValueError):
            continue
        rings = geometry.get("coordinates") or []
        if _point_in_ring_set(lon, lat, rings):
            matched_densities.append(rec.density)

    if matched_densities:
        density = (
            "heavy"
            if "heavy" in matched_densities
            else ("medium" if "medium" in matched_densities else "light")
        )
        return {
            "status": "present",
            "density": density,
            "details": f"Location is inside HMS overhead smoke plume ({density} density)",
        }
    return {
        "status": "absent",
        "density": None,
        "details": "No overhead HMS smoke plume detected at this location",
    }


def _openaq_sig_from_aqs(aqs_records: List[AqsDailyRecord], date_local: str) -> Dict:
    """Build the ``openaq_concentrations`` signal from the day's AQS rows.

    The live OpenAQ signal reports physical concentrations at a nearby monitor;
    the archive's nearest equivalent is the site's own daily AQS summary for
    the same parameters. When a parameter has multiple POC rows, the row with
    the MAX non-null AQI wins (ties broken by highest concentration) — the same
    max-AQI logic ``build_observation`` uses; when no row has an AQI, the row
    with the highest concentration wins. PM2.5 (88101) and PM10 (81102) carry
    through with their reported units; O3/NO2/SO2 (44201/42602/42401) are
    normalized to ppb (×1000 when stored in ppm) and CO (42101) is carried as
    ppm — the units ``score_hypotheses`` compares against ``OPENAQ_*``
    thresholds.
    """
    def _pick(rows: List[AqsDailyRecord]) -> Optional[AqsDailyRecord]:
        """Best row for one parameter: max non-null AQI (tie-break: max
        concentration), falling back to max concentration when no row has an
        AQI. Rows carry a non-None concentration by construction below; an
        empty group yields None (parameter not reported that day)."""
        if not rows:
            return None
        with_aqi = [r for r in rows if r.aqi is not None]
        if with_aqi:
            return max(with_aqi, key=lambda r: (r.aqi, r.concentration or 0.0))
        return max(rows, key=lambda r: r.concentration or 0.0)

    by_param: Dict[str, List[AqsDailyRecord]] = {}
    for rec in aqs_records:
        if rec.concentration is None:
            continue
        by_param.setdefault(rec.parameter_code, []).append(rec)

    pm25_rec = _pick(by_param.get("88101", []))
    pm10_rec = _pick(by_param.get("81102", []))
    o3_rec = _pick(by_param.get("44201", []))
    no2_rec = _pick(by_param.get("42602", []))
    so2_rec = _pick(by_param.get("42401", []))
    co_rec = _pick(by_param.get("42101", []))

    pm25 = pm25_rec.concentration if pm25_rec is not None else None
    pm10 = pm10_rec.concentration if pm10_rec is not None else None
    o3_raw = o3_rec.concentration if o3_rec is not None else None
    o3_units = o3_rec.units if o3_rec is not None else None
    no2_raw = no2_rec.concentration if no2_rec is not None else None
    no2_units = no2_rec.units if no2_rec is not None else None
    so2_raw = so2_rec.concentration if so2_rec is not None else None
    so2_units = so2_rec.units if so2_rec is not None else None
    co_ppm = co_rec.concentration if co_rec is not None else None

    def _to_ppb(value: Optional[float], units: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        if units is not None and "ppm" in units.lower():
            return round(value * 1000.0, 2)
        return value

    o3_ppb = _to_ppb(o3_raw, o3_units)
    no2_ppb = _to_ppb(no2_raw, no2_units)
    so2_ppb = _to_ppb(so2_raw, so2_units)

    if pm25 is None and pm10 is None:
        return {
            "id": "openaq_concentrations",
            "label": "Local Monitor Concentrations (OpenAQ)",
            "status": "unavailable",
            "details": "No PM concentration records for date",
        }

    pm25_pm10_ratio = None
    if pm25 is not None and pm10 is not None and pm10 > 0:
        pm25_pm10_ratio = round(pm25 / pm10, 2)

    return {
        "id": "openaq_concentrations",
        "label": "Local Monitor Concentrations (OpenAQ)",
        "status": "present",
        "pm25": pm25,
        "pm10": pm10,
        "o3_ppb": o3_ppb,
        "no2_ppb": no2_ppb,
        "co_ppm": co_ppm,
        "so2_ppb": so2_ppb,
        "pm25_pm10_ratio": pm25_pm10_ratio,
        "monitor": None,
        "as_of": date_local,
        "daily_percentile": None,
        "same_hour_percentile": None,
        "same_hour_median": None,
        "details": f"AQS daily summary concentrations for {date_local}",
    }


# ---------------------------------------------------------------------------
# Site-day reconstruction + scoring
# ---------------------------------------------------------------------------


def _site_coords(
    aqs_records: List[AqsDailyRecord],
    weather_rec: Optional[WeatherDailyRecord],
) -> Tuple[Optional[float], Optional[float]]:
    """Site coordinates from the day's AQS rows, falling back to the weather
    record. Returns (None, None) when neither source carries coordinates."""
    for rec in aqs_records:
        if rec.lat is not None and rec.lon is not None:
            return rec.lat, rec.lon
    if weather_rec is not None:
        return weather_rec.lat, weather_rec.lon
    return None, None


def reconstruct_signals(
    store: AccuracyStore, site_id: str, date_local: str
) -> Tuple[Dict, List[Dict]]:
    """Rebuild the production evidence inputs for one stored site-day.

    Loads the day's AQS/weather/FIRMS/HMS records from ``store`` and returns
    ``(observation, signals)`` exactly as the engine's ``iter_evidence_signals``
    would yield for the same day — with the historical gaps (AOD, WFIGS, Census
    place context, news) represented as unavailable feeds so scoring treats them
    as outages rather than verified absences.
    """
    aqs_records = store.fetch_aqs_daily(site_id=site_id, date_local=date_local)
    observation = _observation_dict(build_observation(aqs_records))

    weather_records = [
        r for r in store.fetch_weather_daily(site_id=site_id) if r.date_local == date_local
    ]
    weather_rec = weather_records[0] if weather_records else None
    weather_dict = _weather_dict(weather_rec)

    site_lat, site_lon = _site_coords(aqs_records, weather_rec)

    aod_res = {
        "status": "unavailable",
        "aod_value": None,
        "density": None,
        "details": "Historical AOD not available",
    }

    wind_speed = weather_rec.wind_max_mph if weather_rec is not None else None
    wind_dir = weather_rec.wind_dir_dominant_deg if weather_rec is not None else None
    radius_mi = firms_search_radius_miles(wind_speed)

    if site_lat is not None and site_lon is not None:
        pixels = _hotspots_for_day(store, site_lat, site_lon, date_local, wind_dir, radius_mi)
        firms_res = _firms_res_from_hotspots(pixels, site_lat, site_lon, wind_dir, radius_mi)
        hms_res = _hms_res_from_store(store, date_local, site_lat, site_lon)
    else:
        firms_res = {
            "status": "absent",
            "hotspots": [],
            "count": 0,
            "total_count": 0,
            "nearest": None,
            "alignment": None,
            "details": "Site coordinates unavailable for FIRMS query",
        }
        hms_res = {
            "status": "unavailable",
            "density": None,
            "details": "Site coordinates unavailable for HMS check",
        }

    wfigs_res = {
        "status": "unavailable",
        "incident": None,
        "count": 0,
        "alignment": None,
        "details": "Historical incident registry not ingested",
    }
    place_res = {
        "status": "unavailable",
        "details": "Population context not available historically",
    }
    openaq_sig = _openaq_sig_from_aqs(aqs_records, date_local)

    signals = build_evidence_signals(
        observation,
        weather_dict,
        aod_res,
        firms_res,
        None,  # no historical news/incident-name search
        openaq_sig,
        hms_res,
        wfigs_res,
        place_res,
    )
    return observation, signals


def score_sample(
    store: AccuracyStore, site_id: str, date_local: str
) -> Tuple[Dict, List[Dict], List[str]]:
    """Reconstruct and score one site-day with the production scorer.

    Returns ``(observation, hypotheses, open_questions)`` where ``hypotheses``
    are the ranked scorer hypotheses (``score_hypotheses`` output) and
    ``open_questions`` the unresolved-evidence questions.
    """
    observation, signals = reconstruct_signals(store, site_id, date_local)
    hypotheses, open_questions = score_hypotheses(observation, signals)
    return observation, hypotheses, open_questions
