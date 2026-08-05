from typing import Dict, Any, List, Optional, Tuple
from backend.services.openaq import monitor_source_label

# OpenAQ concentration gates. These only apply when the openaq_concentrations
# signal status is "present" (fresh, complete, US reference-monitor data);
# otherwise scoring behaves exactly as before.
OPENAQ_DUST_RATIO_MAX = 0.35       # PM2.5/PM10 ratio below this -> coarse-dominated (dust)
OPENAQ_SMOKE_RATIO_MIN = 0.70      # PM2.5/PM10 ratio at/above this -> fine-dominated (smoke/combustion)
OPENAQ_NO2_PPB = 50.0              # traffic/combustion tracer
OPENAQ_SO2_PPB = 75.0              # industrial point-source tracer
OPENAQ_CO_PPM = 2.0                # combustion tracer
OPENAQ_O3_PPB = 70.0               # EPA 1-hour NAAQS for ground-level ozone
OPENAQ_SAME_HOUR_PERCENTILE = 90.0
OPENAQ_SAME_HOUR_FACTOR = 2.0

# AirNow/OpenAQ disagreement gate. The reported AQI is a longer-term average
# (NowCast) while OpenAQ is an hourly reading from a monitor up to 25 km away.
# If the measured concentration is below half the lower bound of the AQI band
# the reported AQI implies, the readings conflict too strongly to use the
# monitor value as scoring evidence; it is disclosed as an open question
# instead of boosting any hypothesis.
OPENAQ_MEASURED_CONFLICT_FACTOR = 0.5
# Current EPA 24-hour AQI breakpoints (AQS code table "aqi_breakpoints").
# The PM2.5 Good band is now 0.0-9.0 µg/m³ after the tightened annual
# standard; the upper bands tightened to 125.5/225.5/325.5. PM10 is
# unchanged, with a single HAZARDOUS band at 301-500 (floor 425.0).
# Values are the low concentration of each AQI band.
PM25_AQI_LOWER_BOUNDS = (
    (51, 9.1), (101, 35.5), (151, 55.5), (201, 125.5), (301, 225.5), (401, 325.5)
)
PM10_AQI_LOWER_BOUNDS = (
    (51, 55.0), (101, 155.0), (151, 255.0), (201, 355.0), (301, 425.0)
)


def _aqi_concentration_lower_bound(aqi: int, bounds) -> Optional[float]:
    """Lowest concentration (µg/m³) the reported AQI band implies."""
    lower = None
    for band_aqi, concentration in bounds:
        if aqi >= band_aqi:
            lower = concentration
    return lower


def score_hypotheses(
    observation: Dict[str, Any],
    signals: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Deterministically score the five explicit attribution hypotheses based on observable signal criteria:
    1. wildfire_smoke
    2. ozone_episode
    3. windblown_dust
    4. winter_stagnation
    5. urban_industrial_pm

    Returns (ranked_hypotheses, open_questions).
    """
    sig_map = {s["id"]: s for s in signals}
    
    aod = sig_map.get("aerosol_plume", {})
    firms = sig_map.get("firms_upwind", {})
    wind = sig_map.get("wind", {})
    pm = sig_map.get("surface_pm_level", {})
    o3 = sig_map.get("ozone_heat", {})

    aqi_val = observation.get("aqi", 0)
    primary = observation.get("primary_pollutant", "").upper()

    hypotheses = []
    open_questions = []

    # Extract boolean & numeric signal criteria
    aod_present = aod.get("status") == "present"
    aod_value = float(aod.get("aod_value") or 0.0)
    
    firms_present = firms.get("status") == "present"
    firms_count = firms.get("count", 0)
    nearest_firm = firms.get("nearest")
    incident_name = firms.get("incident_name")
    firms_alignment = firms.get("alignment")  # "upwind" | "nearby" | None

    wind_speed = wind.get("speed_mph")
    boundary_layer_height_m = wind.get("boundary_layer_height_m")

    pm_primary = pm.get("primary", False)
    pm10_primary = pm.get("pm10_primary", False) or ("PM10" in primary)
    pm_elevated = pm.get("elevated", False) or (aqi_val > 50 and pm_primary)

    temp_f = o3.get("temperature_f")
    is_calm = wind_speed is not None and wind_speed < 4.0
    is_high_wind = wind_speed is not None and wind_speed >= 15.0
    is_cold = temp_f is not None and temp_f < 55.0

    # OpenAQ reference-monitor concentrations (gated on status == "present").
    openaq = sig_map.get("openaq_concentrations", {})
    op_present = openaq.get("status") == "present"
    pm25_conc = openaq.get("pm25")
    o3_ppb = openaq.get("o3_ppb")
    no2_ppb = openaq.get("no2_ppb")
    so2_ppb = openaq.get("so2_ppb")
    co_ppm = openaq.get("co_ppm")
    pm_ratio = openaq.get("pm25_pm10_ratio")
    same_hour_pct = openaq.get("same_hour_percentile")
    same_hour_median = openaq.get("same_hour_median")
    daily_pct = openaq.get("daily_percentile")
    openaq_monitor = openaq.get("monitor") or {}
    openaq_dist_km = openaq_monitor.get("distance_km")
    openaq_label = monitor_source_label(openaq_monitor)

    # Conflict gate: elevated reported AQI but a much lower measured reading.
    measured_conflict = False
    conflict_value = None
    if op_present and pm_elevated:
        if pm_primary and not pm10_primary:
            bounds, conflict_value = PM25_AQI_LOWER_BOUNDS, pm25_conc
        elif pm10_primary:
            bounds, conflict_value = PM10_AQI_LOWER_BOUNDS, openaq.get("pm10")
        else:
            bounds, conflict_value = None, None
        if bounds and conflict_value is not None:
            lower = _aqi_concentration_lower_bound(aqi_val, bounds)
            if lower is not None and conflict_value < lower * OPENAQ_MEASURED_CONFLICT_FACTOR:
                measured_conflict = True

    measured_line = None
    if pm25_conc is not None:
        measured_line = (
            f"PM2.5 measured at {pm25_conc:.0f} micrograms per cubic meter "
            f"at the {openaq_label}"
        )
        if openaq_dist_km is not None:
            measured_line += f" {openaq_dist_km:.0f} km from here"

    monitor_distance_suffix = (
        f" at the nearest reporting monitor {openaq_dist_km:.0f} km away"
        if openaq_dist_km is not None else " at the nearest reporting monitor"
    )

    fine_dominated = (
        op_present and not measured_conflict and pm_ratio is not None
        and pm_ratio >= OPENAQ_SMOKE_RATIO_MIN
    )
    coarse_dominated = (
        op_present and not measured_conflict and pm_ratio is not None
        and pm_ratio < OPENAQ_DUST_RATIO_MAX
    )
    urban_tracer = op_present and not measured_conflict and pm_elevated and (
        (no2_ppb is not None and no2_ppb >= OPENAQ_NO2_PPB)
        or (so2_ppb is not None and so2_ppb >= OPENAQ_SO2_PPB)
        or (co_ppm is not None and co_ppm >= OPENAQ_CO_PPM)
    )
    same_hour_anomaly = (
        op_present and not measured_conflict and pm_elevated and pm25_conc is not None
        and same_hour_pct is not None and same_hour_pct >= OPENAQ_SAME_HOUR_PERCENTILE
        and same_hour_median is not None and same_hour_median > 0
        and pm25_conc >= OPENAQ_SAME_HOUR_FACTOR * same_hour_median
    )

    # News may decorate FIRMS/AOD evidence, but only FIRMS makes a news name a fire vote
    # (AOD-only + news was enough to falsely crown "Man Starts Fire" as high-confidence smoke).
    has_firms_corroboration = firms_present and firms_count > 0
    firms_upwind = has_firms_corroboration and firms_alignment != "nearby"
    has_haze = aod_present and aod_value >= 0.2
    aod_only_fire = (aod_value >= 0.4) and not has_firms_corroboration
    # Light haze + Very Unhealthy+ PM: soft smoke support (Burns-class) without the hard 0.4 cliff.
    # Clear AOD + extreme PM stays local (food-truck / structure burn) — no soft smoke vote.
    light_haze_extreme_pm = (
        has_haze and aod_value < 0.4 and aqi_val >= 150 and pm_elevated and not has_firms_corroboration
    )

    # Count verified positive fire evidence indicators
    positive_fire_signals = []
    if aod_present and aod_value >= 0.4:
        positive_fire_signals.append(f"Dense atmospheric column particle plume detected (AOD {aod_value:.2f})")
    elif light_haze_extreme_pm:
        positive_fire_signals.append(
            f"Light atmospheric haze (AOD {aod_value:.2f}) with very unhealthy surface PM (AQI {aqi_val}), consistent with settling smoke"
        )
    if has_firms_corroboration:
        loc_str = f" ({nearest_firm['distance_miles']} mi {nearest_firm['bearing']})" if nearest_firm else ""
        align_label = "upwind" if firms_upwind else "nearby (not wind-aligned)"
        positive_fire_signals.append(
            f"Detected {firms_count} active NASA FIRMS thermal hotspot cluster(s) {align_label}{loc_str}"
        )

    # News alone is NOT a fire vote. Only count news when FIRMS hotspots corroborate.
    if incident_name and has_firms_corroboration:
        positive_fire_signals.append(f"Recent news mention of '{incident_name}'")

    fire_signal_count = len(positive_fire_signals)

    # Unverified / non-FIRMS news mentions become open questions, not votes
    if incident_name and not has_firms_corroboration:
        if has_haze:
            open_questions.append(
                f"Recent news mentions '{incident_name}', but no current upwind hotspots; haze may be regional/urban aerosol rather than that incident."
            )
        else:
            open_questions.append(
                f"Recent news mentions '{incident_name}', but no current upwind hotspots or atmospheric haze; may be extinguished, distant, or unrelated."
            )

    # -------------------------------------------------------------
    # Mode 1: Wildfire Smoke Transport
    # -------------------------------------------------------------
    smoke_support = list(positive_fire_signals)
    smoke_against = []

    if pm_primary and pm_elevated and not light_haze_extreme_pm:
        # light_haze_extreme_pm already mentions AQI in its fire signal
        smoke_support.append(f"Surface PM2.5/PM10 is elevated as primary pollutant (AQI {aqi_val})")

    if fine_dominated and pm_elevated and (aod_present or has_firms_corroboration):
        smoke_support.append(
            f"PM is fine-particle dominated{monitor_distance_suffix}, consistent with smoke rather than coarse dust"
        )

    if (
        op_present and pm_elevated and pm25_conc is not None
        and not measured_conflict and (aod_present or has_firms_corroboration)
    ):
        smoke_support.append(measured_line)

    if coarse_dominated and pm_elevated:
        smoke_against.append(
            f"PM is coarse-particle dominated{monitor_distance_suffix}, favoring dust over smoke"
        )

    # Contradiction: Heavy column plume present but surface PM not elevated (Aloft smoke)
    if aod_value >= 0.4 and not pm_elevated:
        smoke_against.append(f"High atmospheric particle density overhead (AOD {aod_value:.2f}), but surface monitors show low ground PM levels")
        open_questions.append("Atmospheric particle plume is present overhead (aloft), but has not settled down to ground breathing level.")

    # Clean AOD vs elevated PM: clear column overhead favors local surface sources
    if pm_elevated and (not aod_present or aod_value < 0.2):
        smoke_against.append("Clear column overhead (clean AOD) while surface PM is elevated, favoring localized surface sources rather than smoke transport")

    if firms.get("status") == "absent" and not aod_present and not incident_name:
        smoke_against.append("No active upwind fires or dense atmospheric plumes detected")
    elif aod_only_fire and pm_elevated:
        smoke_against.append("No nearby upwind thermal hotspots; overhead haze may be urban/regional aerosol rather than verified wildfire smoke")
    elif light_haze_extreme_pm:
        smoke_against.append("No nearby thermal hotspots yet; light haze + extreme PM is suggestive but not hotspot-verified")

    # Deterministic Confidence Classification
    # High requires nearby FIRMS — AOD+news alone must not crown wildfire.
    if aod_value >= 0.4 and not pm_elevated:
        smoke_conf = "low"
        smoke_score = 30
    elif firms_upwind and fire_signal_count >= 2 and pm_elevated:
        smoke_conf = "high"
        smoke_score = 90
    elif firms_upwind and pm_elevated:
        smoke_conf = "high" if aod_value >= 0.8 else "medium"
        smoke_score = 75 if aod_value >= 0.8 else 65
    elif has_firms_corroboration and pm_elevated:
        # Nearby (non-upwind) hotspots — solid regional fire evidence, slightly weaker than upwind
        smoke_conf = "medium"
        smoke_score = 70 if aod_value >= 0.2 else 60
    elif aod_value >= 0.8 and pm_elevated:
        # Heavy plume, no nearby hotspots — possible long-range transport
        smoke_conf = "medium"
        smoke_score = 70
    elif light_haze_extreme_pm:
        # Burns-class: light AOD + Very Unhealthy+ PM without FIRMS
        smoke_conf = "medium"
        smoke_score = 65
    elif aod_value >= 0.4 and pm_elevated:
        # Medium haze without FIRMS — possible smoke, but often urban/regional aerosol
        smoke_conf = "medium"
        smoke_score = 55
    elif fire_signal_count >= 1:
        smoke_conf = "medium"
        smoke_score = 50
    else:
        smoke_conf = "low"
        smoke_score = 25

    if fine_dominated and pm_elevated and (aod_present or has_firms_corroboration):
        smoke_score = max(smoke_score, 60)
        if smoke_conf == "low":
            smoke_conf = "medium"
    if (
        op_present and not measured_conflict and pm_elevated and co_ppm is not None
        and co_ppm >= OPENAQ_CO_PPM and (aod_present or has_firms_corroboration)
    ):
        smoke_support.append("Elevated carbon monoxide alongside PM, a combustion signature consistent with smoke")
        smoke_score = max(smoke_score, 55)
        if smoke_conf == "low":
            smoke_conf = "medium"

    place_desc = None
    if nearest_firm:
        align_word = "Upwind" if firms_upwind else "Nearby"
        place_desc = (
            f"Incident: {incident_name}"
            if (incident_name and has_firms_corroboration)
            else f"{align_word} hotspots {nearest_firm['distance_miles']} mi {nearest_firm['bearing']}"
        )
    elif incident_name and has_firms_corroboration:
        place_desc = f"Incident: {incident_name}"

    hypotheses.append({
        "id": "wildfire_smoke",
        "title": "Wildfire Smoke Transport",
        "score": smoke_score,
        "confidence": smoke_conf,
        "support": smoke_support,
        "against": smoke_against,
        "place": {
            "bearing": nearest_firm.get("bearing") if nearest_firm else None,
            "approx_km": nearest_firm.get("distance_km") if nearest_firm else None,
            "description": place_desc or "Regional smoke transport"
        } if (nearest_firm or (incident_name and has_firms_corroboration)) else None
    })

    # -------------------------------------------------------------
    # Mode 2: Photochemical Ozone Episode
    # -------------------------------------------------------------
    o3_support = []
    o3_against = []

    o3_primary = "O3" in primary or "OZONE" in primary
    is_hot = o3.get("hot_day", False)

    if o3_primary:
        o3_support.append(f"Ground-level Ozone (O3) is the primary reporting pollutant (AQI {aqi_val})")
    else:
        o3_against.append(f"Primary pollutant is {primary}, not Ozone")

    if is_hot:
        o3_support.append(f"High ambient temperature ({temp_f}°F) promotes photochemical ozone formation")
    elif temp_f is not None and temp_f < 75.0:
        o3_against.append(f"Cool ambient temperature ({temp_f}°F) suppresses rapid photochemical ozone generation")

    if o3_primary and is_hot:
        o3_conf = "high"
        o3_score = 85
    elif o3_primary:
        o3_conf = "medium"
        o3_score = 60
    elif is_hot and aqi_val > 50:
        o3_conf = "medium"
        o3_score = 45
    else:
        o3_conf = "low"
        o3_score = 15

    # Measured monitor ozone corroborates the ozone hypothesis (EPA 1-hour
    # standard, 70 ppb). No conflict gate here: a low current 1-hour reading
    # does not contradict an 8-hour-based ozone AQI, so only the elevated
    # direction is scored.
    if op_present and o3_primary and o3_ppb is not None and o3_ppb >= OPENAQ_O3_PPB:
        o3_support.append(
            f"Monitor data shows elevated ground-level ozone ({o3_ppb:.0f} parts per billion) "
            f"at the {openaq_label}"
            + (f" {openaq_dist_km:.0f} km from here" if openaq_dist_km is not None else "")
        )
        o3_score = max(o3_score, 70)

    hypotheses.append({
        "id": "ozone_episode",
        "title": "Photochemical Ozone Episode",
        "score": o3_score,
        "confidence": o3_conf,
        "support": o3_support,
        "against": o3_against
    })

    # -------------------------------------------------------------
    # Mode 3: Windblown Dust Storm
    # -------------------------------------------------------------
    dust_support = []
    dust_against = []

    if pm10_primary:
        dust_support.append("PM10 (coarse particulate) is the primary reported pollutant, consistent with wind-driven dust rather than combustion smoke")
    else:
        dust_against.append(f"Primary pollutant is {primary}, not coarse PM10; dust is not the dominant particle fraction")

    if is_high_wind:
        dust_support.append(f"Sustained wind speed ({wind_speed} mph) is high enough to loft soil and dust")
    elif wind_speed is not None:
        dust_against.append(f"Wind speed ({wind_speed} mph) is below the threshold typically needed to loft significant dust")

    if coarse_dominated and pm_elevated:
        dust_support.append(
            f"PM2.5 is only a small fraction of PM10{monitor_distance_suffix}, consistent with windblown dust"
        )

    if pm10_primary and is_high_wind:
        dust_conf, dust_score = "high", 85
    elif pm10_primary and wind_speed is not None and wind_speed >= 8.0:
        dust_conf, dust_score = "medium", 55
    elif pm10_primary:
        dust_conf, dust_score = "low", 30
    else:
        dust_conf, dust_score = "low", 10

    if coarse_dominated and pm_elevated:
        dust_score = max(dust_score, 55)
        if dust_conf == "low":
            dust_conf = "medium"

    hypotheses.append({
        "id": "windblown_dust",
        "title": "Windblown Dust Storm",
        "score": dust_score,
        "confidence": dust_conf,
        "support": dust_support,
        "against": dust_against
    })

    # -------------------------------------------------------------
    # Mode 4: Winter Stagnation & Temperature Inversion
    # -------------------------------------------------------------
    stagnation_support = []
    stagnation_against = []

    if pm_elevated:
        stagnation_support.append(f"Surface PM2.5 is elevated (AQI {aqi_val})")
    if is_cold:
        stagnation_support.append(f"Cold surface temperature ({temp_f}°F) is consistent with a trapping winter inversion")
    elif temp_f is not None:
        stagnation_against.append(f"Ambient temperature ({temp_f}°F) is not cold enough to be a typical winter stagnation event")
    if is_calm:
        stagnation_support.append(f"Calm winds ({wind_speed} mph) allow pollutants to accumulate near the surface")
    elif wind_speed is not None:
        stagnation_against.append(f"Wind speed ({wind_speed} mph) is high enough to disperse pollutants, arguing against stagnation")
    if fire_signal_count == 0:
        stagnation_support.append("No upwind fire evidence; elevated PM is more likely local combustion/traffic trapped near the surface")
    else:
        stagnation_against.append("Active fire evidence present; elevated PM may be smoke-related rather than pure stagnation")
    if boundary_layer_height_m is not None and boundary_layer_height_m < 500:
        stagnation_support.append(f"Shallow boundary layer height ({boundary_layer_height_m:.0f}m) indicates trapped near-surface air")

    if same_hour_anomaly:
        if is_cold and is_calm:
            stagnation_support.append(
                "PM2.5 is well above the usual reading for this time of day, consistent with air being trapped near the surface"
            )
        else:
            open_questions.append(
                "PM2.5 is unusually high for this time of day, but current conditions don't point to a trapping inversion"
            )

    if pm_elevated and is_cold and is_calm and fire_signal_count == 0:
        stagnation_conf = "high" if (boundary_layer_height_m is not None and boundary_layer_height_m < 500) else "medium"
        stagnation_score = 85 if stagnation_conf == "high" else 65
    elif pm_elevated and is_cold and is_calm:
        stagnation_conf, stagnation_score = "low", 30
    else:
        stagnation_conf, stagnation_score = "low", 15

    if same_hour_anomaly and is_cold and is_calm:
        stagnation_score = max(stagnation_score, 70)
        if stagnation_conf == "low":
            stagnation_conf = "medium"

    hypotheses.append({
        "id": "winter_stagnation",
        "title": "Winter Stagnation & Temperature Inversion",
        "score": stagnation_score,
        "confidence": stagnation_conf,
        "support": stagnation_support,
        "against": stagnation_against
    })

    # -------------------------------------------------------------
    # Mode 5: Localized Urban / Industrial PM
    # -------------------------------------------------------------
    # Demote urban only when haze/FIRMS support regional smoke — never for high AQI alone
    # (extreme local burns / "1000 food trucks" with clear AOD stay urban/local).
    urban_support = []
    urban_against = []

    is_dust_or_stagnation = pm10_primary or (is_cold and is_calm)

    if pm_elevated and fire_signal_count == 0 and not is_dust_or_stagnation:
        urban_support.append(f"PM2.5 is elevated (AQI {aqi_val}) without verified fire, dust, or stagnation signals")
    elif pm_elevated and light_haze_extreme_pm and not is_dust_or_stagnation:
        urban_support.append(
            f"PM2.5 is very elevated (AQI {aqi_val}), but light haze suggests regional smoke may contribute alongside local sources"
        )
    elif pm_elevated and aod_only_fire and not is_dust_or_stagnation:
        urban_support.append(
            f"PM2.5 is elevated (AQI {aqi_val}) with regional haze but no nearby fire hotspots; local urban sources remain a strong explanation"
        )
    elif pm_elevated:
        urban_support.append("PM is elevated, but a more specific mode (fire, dust, or stagnation) better explains it")

    if urban_tracer:
        parts = []
        if no2_ppb is not None and no2_ppb >= OPENAQ_NO2_PPB:
            parts.append("traffic-related nitrogen dioxide")
        if so2_ppb is not None and so2_ppb >= OPENAQ_SO2_PPB:
            parts.append("industrial sulfur dioxide")
        if co_ppm is not None and co_ppm >= OPENAQ_CO_PPM:
            parts.append("combustion-related carbon monoxide")
        urban_support.append(
            "Monitor data shows elevated " + " and ".join(parts) + ", a signature of local combustion and traffic"
        )
    if fine_dominated and pm_elevated and not (aod_present or has_firms_corroboration):
        urban_support.append(
            f"PM is fine-particle dominated{monitor_distance_suffix}, consistent with local combustion or traffic rather than coarse dust"
        )

    if (
        op_present and pm_elevated and pm25_conc is not None
        and not measured_conflict and not (aod_present or has_firms_corroboration)
    ):
        urban_support.append(measured_line)

    if not pm_elevated:
        urban_against.append("Surface PM is within satisfactory/normal range")
    if has_firms_corroboration:
        urban_against.append(f"Verified fire evidence signals present ({fire_signal_count} indicator(s))")
    elif light_haze_extreme_pm:
        urban_against.append(
            "Light atmospheric haze with very unhealthy PM, favoring regional smoke settling over purely local urban emissions"
        )
    elif fire_signal_count > 0 and aod_value >= 0.8:
        urban_against.append("Heavy atmospheric particle plume present; long-range smoke may contribute")
    if is_dust_or_stagnation:
        urban_against.append("Conditions better match windblown dust or winter stagnation than generic urban/industrial PM")

    if pm_elevated and fire_signal_count == 0 and not is_dust_or_stagnation:
        # Clear AOD + elevated PM (any AQI) → local/urban wins — food-truck safe
        urban_conf, urban_score = "high", 75
    elif pm_elevated and light_haze_extreme_pm and not is_dust_or_stagnation:
        # Burns-class: demote urban so soft smoke can rank above
        urban_conf, urban_score = "medium", 40
    elif pm_elevated and aod_only_fire and aod_value < 0.8 and not is_dust_or_stagnation:
        # Medium haze, no FIRMS — urban stays competitive with AOD-only smoke (score 55)
        urban_conf, urban_score = "high", 70
    elif pm_elevated and aod_only_fire and aod_value >= 0.8 and not is_dust_or_stagnation:
        # Heavy plume without nearby hotspots — urban competes but does not dominate
        urban_conf, urban_score = "medium", 50
    elif pm_elevated:
        urban_conf, urban_score = "medium", 35
    else:
        urban_conf, urban_score = "low", 15

    if urban_tracer:
        urban_score = max(urban_score, 55)
        if urban_conf == "low":
            urban_conf = "medium"
    if fine_dominated and pm_elevated and not (aod_present or has_firms_corroboration):
        urban_score = max(urban_score, 55)
        if urban_conf == "low":
            urban_conf = "medium"

    hypotheses.append({
        "id": "urban_industrial_pm",
        "title": "Localized Urban / Industrial PM",
        "score": urban_score,
        "confidence": urban_conf,
        "support": urban_support,
        "against": urban_against
    })

    # Sort hypotheses by score descending
    hypotheses.sort(key=lambda h: h["score"], reverse=True)

    if aqi_val <= 50:
        open_questions.append("Air quality index is currently in the 'Good' range (AQI ≤ 50).")
    if op_present and not measured_conflict and pm_elevated and daily_pct is not None and daily_pct >= 90:
        open_questions.append(
            "PM2.5 is well above this location's typical daily readings, an unusual day for this area"
        )
    # Only disclose the mismatch when the monitor is close enough to be
    # meaningful (or distance is unknown); a distant low reading is spatial
    # variation and explaining it just confuses users.
    if (
        measured_conflict and conflict_value is not None
        and (openaq_dist_km is None or openaq_dist_km <= 10.0)
    ):
        category = observation.get("category", "elevated")
        dist_part = f" ({openaq_dist_km:.0f} km away)" if openaq_dist_km is not None else ""
        open_questions.append(
            f"The reported AQI is {category} ({aqi_val}), but the nearest air quality monitor "
            f"measures {conflict_value:.0f} micrograms per cubic meter right now{dist_part}; the AQI is "
            "a longer-term average while the monitor reading is current, so the two can differ"
        )
    return hypotheses, open_questions
