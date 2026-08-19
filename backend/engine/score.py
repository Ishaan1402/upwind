from typing import Dict, Any, List, Optional, Tuple
from backend.services.openaq import monitor_source_label
from backend.engine.params import get_params


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
    p = get_params()
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
    # A feed outage shouldn't conclude as a verified absence
    firms_absent = firms.get("status") == "absent"
    firms_unavailable = firms.get("status") == "unavailable"

    wind_speed = wind.get("speed_mph")
    boundary_layer_height_m = wind.get("boundary_layer_height_m")

    pm_primary = pm.get("primary", False)
    pm10_primary = pm.get("pm10_primary", False) or ("PM10" in primary)
    # None-safe: a missing AQI is unknown, never elevated.
    pm_elevated = pm.get("elevated", False) or (
        aqi_val is not None and aqi_val > p.aqi_elevated and pm_primary
    )

    temp_f = o3.get("temperature_f")
    is_calm = wind_speed is not None and wind_speed < p.calm_wind_speed_mph
    is_high_wind = wind_speed is not None and wind_speed >= p.high_wind_speed_mph
    is_cold = temp_f is not None and temp_f < p.cold_temp_f

    # OpenAQ reference monitor concentrations
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

    # Conflict gate for elevated reported AQI vs lower OpenAQ measured reading
    measured_conflict = False
    conflict_value = None
    if op_present and pm_elevated:
        if pm_primary and not pm10_primary:
            bounds, conflict_value = p.pm25_aqi_lower_bounds, pm25_conc
        elif pm10_primary:
            bounds, conflict_value = p.pm10_aqi_lower_bounds, openaq.get("pm10")
        else:
            bounds, conflict_value = None, None
        if bounds and conflict_value is not None:
            lower = _aqi_concentration_lower_bound(aqi_val, bounds)
            if lower is not None and conflict_value < lower * p.openaq_measured_conflict_factor:
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
        and pm_ratio >= p.openaq_smoke_ratio_min
    )
    coarse_dominated = (
        op_present and not measured_conflict and pm_ratio is not None
        and pm_ratio < p.openaq_dust_ratio_max
    )
    # No fine/coarse ratio available (PM10-only/PM2.5-only monitors, or OpenAQ
    # entirely down): the smoke-vs-dust discriminator is genuinely unavailable.
    ratio_missing = pm_ratio is None
    urban_tracer = op_present and not measured_conflict and pm_elevated and (
        (no2_ppb is not None and no2_ppb >= p.openaq_no2_ppb)
        or (so2_ppb is not None and so2_ppb >= p.openaq_so2_ppb)
        or (co_ppm is not None and co_ppm >= p.openaq_co_ppm)
    )
    same_hour_anomaly = (
        op_present and not measured_conflict and pm_elevated and pm25_conc is not None
        and same_hour_pct is not None and same_hour_pct >= p.openaq_same_hour_percentile
        and same_hour_median is not None and same_hour_median > 0
        and pm25_conc >= p.openaq_same_hour_factor * same_hour_median
    )

    # News requires verified satellite record or WFIGS to count as a fire vote
    has_firms_corroboration = firms_present and firms_count > 0

    hms_sig = sig_map.get("hms_smoke", {})
    wfigs_sig = sig_map.get("wfigs_incident", {})
    hms_present = hms_sig.get("status") == "present"
    hms_density = hms_sig.get("density")
    wfigs_present = wfigs_sig.get("status") == "present"
    wfigs_incident = wfigs_sig.get("incident")
    wfigs_alignment = wfigs_sig.get("alignment")

    # Corroboration across verified feeds (FIRMS, HMS, or upwind/nearby WFIGS)
    wfigs_corroborates = wfigs_present and (
        wfigs_alignment == "upwind"
        or (
            wfigs_incident is not None
            and isinstance(wfigs_incident.get("distance_miles"), (int, float))
            and wfigs_incident["distance_miles"] <= p.wfigs_corroboration_radius_miles
        )
    )
    has_smoke_corroboration = has_firms_corroboration or hms_present or wfigs_corroborates
    firms_upwind = has_firms_corroboration and firms_alignment != "nearby"
    has_haze = aod_present and aod_value >= p.aod_haze
    # Plume with verified absence of hotspots
    aod_only_fire = (aod_value >= p.aod_medium) and firms_absent and not has_smoke_corroboration
    # Plume without verified fire corroboration
    haze_no_fire = (aod_value >= p.aod_medium) and not has_smoke_corroboration
    # Light haze with extreme PM indicates possible regional smoke settling
    light_haze_extreme_pm = (
        has_haze and aod_value < p.aod_medium
        and aqi_val is not None and aqi_val >= p.extreme_pm_aqi
        and pm_elevated and not has_smoke_corroboration
    )

    # Count verified positive fire evidence indicators
    positive_fire_signals = []
    # AOD is modeled column loading, NOT fire evidence; it corroborates but is never a fire vote.
    if has_firms_corroboration:
        loc_str = f" ({nearest_firm['distance_miles']} mi {nearest_firm['bearing']})" if nearest_firm else ""
        # The alignment word describes the NAMED cluster (nearest), not the
        # upwind subset that firms_upwind gates: 'nearest' may be the strongest
        # overall cluster, which can be downwind even when upwind clusters exist.
        align_label = "upwind" if (nearest_firm or {}).get("is_upwind") else "nearby (not wind-aligned)"
        positive_fire_signals.append(
            f"Detected {firms_count} active NASA FIRMS thermal hotspot cluster(s) {align_label}{loc_str}"
        )
    if hms_present:
        density_label = hms_density or "light"
        positive_fire_signals.append(
            f"NOAA smoke-plume analysis places this location inside a {density_label} smoke plume"
        )
    if wfigs_corroborates and wfigs_incident:
        inc = wfigs_incident
        dist_val = inc.get("distance_miles")
        dist_part = f" {dist_val} mi {inc.get('bearing')}" if isinstance(dist_val, (int, float)) else ""
        meta = []
        if isinstance(inc.get("size_acres"), (int, float)):
            meta.append(f"{inc['size_acres']:,.0f} acres")
        if isinstance(inc.get("percent_contained"), (int, float)):
            meta.append(f"{inc['percent_contained']:.0f}% contained")
        meta_part = f" ({', '.join(meta)})" if meta else ""
        positive_fire_signals.append(
            f"Federal incident registry lists '{inc['name']}'{meta_part}{dist_part}"
        )

    if incident_name and has_smoke_corroboration:
        positive_fire_signals.append(f"Recent news mention of '{incident_name}'")

    fire_signal_count = len(positive_fire_signals)

    # Unverified news mentions become open questions
    if incident_name and not has_smoke_corroboration:
        if has_haze:
            open_questions.append(
                f"Recent news mentions '{incident_name}', but no current upwind hotspots; haze may be regional/urban aerosol rather than that incident."
            )
        else:
            open_questions.append(
                f"Recent news mentions '{incident_name}', but no current upwind hotspots or atmospheric haze; may be extinguished, distant, or unrelated."
            )

    # Mode 1: Wildfire Smoke
    smoke_support = list(positive_fire_signals)
    smoke_against = []

    if pm_primary and pm_elevated and not light_haze_extreme_pm:
        smoke_support.append(f"Surface PM2.5/PM10 is elevated as primary pollutant (AQI {aqi_val})")

    if fine_dominated and pm_elevated and (aod_present or has_smoke_corroboration):
        smoke_support.append(
            f"PM is fine-particle dominated{monitor_distance_suffix}, consistent with smoke rather than coarse dust"
        )

    if (
        op_present and pm_elevated and pm25_conc is not None
        and not measured_conflict and (aod_present or has_smoke_corroboration)
    ):
        smoke_support.append(measured_line)

    if coarse_dominated and pm_elevated:
        smoke_against.append(
            f"PM is coarse-particle dominated{monitor_distance_suffix}, favoring dust over smoke"
        )

    if aod_value >= p.aod_medium and not pm_elevated:
        smoke_against.append(f"High atmospheric particle density overhead (AOD {aod_value:.2f}), but surface monitors show low ground PM levels")
        open_questions.append("Atmospheric particle plume is present overhead (aloft), but has not settled down to ground breathing level.")

    # Aloft smoke with clean surface PM
    if has_smoke_corroboration and not pm_elevated:
        smoke_against.append("Smoke or fire activity is reported, but surface PM is not elevated, so the smoke is likely aloft or regional rather than at breathing level")

    # Clean AOD with elevated surface PM
    if pm_elevated and (not aod_present or aod_value < p.aod_haze) and not has_smoke_corroboration:
        smoke_against.append("Clear column overhead (clean AOD) while surface PM is elevated, favoring localized surface sources rather than smoke transport")

    if firms.get("status") == "absent" and not aod_present and not incident_name and not has_smoke_corroboration:
        smoke_against.append("No active upwind fires or dense atmospheric plumes detected")
    elif aod_only_fire and pm_elevated:
        smoke_against.append("No nearby upwind thermal hotspots; overhead haze may be urban/regional aerosol rather than verified wildfire smoke")
    elif light_haze_extreme_pm and firms_absent:
        smoke_against.append("No nearby thermal hotspots yet; light haze + extreme PM is suggestive but not hotspot-verified")

    # Disclose feed outages as open questions rather than verified absence
    if firms_unavailable and (pm_elevated or aod_present or incident_name):
        open_questions.append(
            "Satellite thermal detection is currently unavailable, so the absence of detected hotspots is not evidence against wildfire smoke."
        )
    if hms_sig.get("status") == "unavailable" and (pm_elevated or aod_present or incident_name):
        open_questions.append(
            "The NOAA smoke-plume analysis feed is currently unavailable, so the absence of a smoke-plume reading is not evidence against wildfire smoke."
        )
    if wfigs_sig.get("status") == "unavailable" and (pm_elevated or aod_present or incident_name):
        open_questions.append(
            "The federal wildfire incident registry is currently unavailable, so the absence of a listed incident is not evidence against wildfire smoke."
        )

    # Deterministic confidence classification
    if aod_value >= p.aod_medium and not pm_elevated:
        smoke_conf = "low"
        smoke_score = 30
    elif has_smoke_corroboration and not pm_elevated:
        # Aloft smoke with clean ground air
        smoke_conf = "low"
        smoke_score = 30
    elif firms_upwind and fire_signal_count >= p.fire_signal_high_count and pm_elevated:
        smoke_conf = "high"
        smoke_score = 90
    elif firms_upwind and pm_elevated:
        smoke_conf = "medium"
        smoke_score = 65
    elif hms_present and pm_elevated:
        # Analyst-verified plume; high only with a dense analyst plume
        hms_high = hms_density in ("medium", "heavy")
        smoke_conf = "high" if hms_high else "medium"
        smoke_score = 85 if hms_high else 70
    elif wfigs_corroborates and wfigs_alignment == "upwind" and pm_elevated:
        # Upwind WFIGS federal incident scoring; only extreme PM pushes to high
        extreme_pm = aqi_val >= p.extreme_pm_aqi
        smoke_conf = "high" if extreme_pm else "medium"
        smoke_score = 80 if extreme_pm else 70
    elif has_firms_corroboration and pm_elevated:
        # Nearby non-upwind FIRMS hotspot scoring
        smoke_conf = "medium"
        smoke_score = 70 if aod_value >= p.aod_haze else 60
    elif wfigs_corroborates and pm_elevated:
        # Nearby non-upwind WFIGS incident scoring
        smoke_conf, smoke_score = "medium", 55
    else:
        smoke_conf = "low"
        smoke_score = 25

    if fine_dominated and pm_elevated and has_smoke_corroboration:
        smoke_score = max(smoke_score, 60)
        if smoke_conf == "low":
            smoke_conf = "medium"
    if (
        op_present and not measured_conflict and pm_elevated and co_ppm is not None
        and co_ppm >= p.openaq_co_ppm and has_smoke_corroboration
    ):
        smoke_support.append("Elevated carbon monoxide alongside PM, a combustion signature consistent with smoke")
        smoke_score = max(smoke_score, 55)
        if smoke_conf == "low":
            smoke_conf = "medium"

    # Coarse-dominated PM10 with elevated surface PM is windblown dust, not
    # smoke. The PM2.5/PM10 ratio is the honest discriminator and it reads
    # coarse; fire evidence alone must not override it. This only fires when the
    # ratio is actually present (coarse_dominated requires pm_ratio is not None),
    # so it never guesses on PM10-only monitors that lack a PM2.5 row.
    if coarse_dominated and pm10_primary and pm_elevated:
        smoke_score = min(smoke_score, 45)
        smoke_conf = "low"

    # Windy PM10-primary days where the fine/coarse ratio is MISSING (PM10-only
    # or PM2.5-only monitors, or OpenAQ entirely down) are dust-suspect: without
    # the ratio smoke cannot be confirmed, and strong wind + coarse-primary PM
    # favors windblown dust. Fire evidence alone must not crown smoke on these days.
    if pm10_primary and pm_elevated and is_high_wind and ratio_missing:
        smoke_score = min(smoke_score, 45)
        smoke_conf = "low"
        smoke_against.append(
            f"The fine/coarse particle ratio is unavailable{monitor_distance_suffix}, so smoke cannot be confirmed against a windblown-dust alternative"
        )
        open_questions.append(
            "PM10 is elevated with strong wind and no fine/coarse particle ratio "
            "available, so smoke cannot be confirmed; windblown dust is the "
            "better-supported explanation."
        )

    place_desc = None
    if nearest_firm:
        # 'nearest' is the top cluster across ALL clusters, so describe the
        # cluster actually named (its own is_upwind flag), not the upwind subset.
        align_word = "Upwind" if nearest_firm.get("is_upwind") else "Nearby"
        place_desc = (
            f"News-reported fire '{incident_name}' (size unknown)"
            if (incident_name and has_firms_corroboration)
            else f"{align_word} hotspots {nearest_firm['distance_miles']} mi {nearest_firm['bearing']}"
        )
    elif incident_name and has_firms_corroboration:
        place_desc = f"News-reported fire '{incident_name}' (size unknown)"
    elif wfigs_corroborates and wfigs_incident:
        place_desc = f"Incident: {wfigs_incident['name']}"

    # Prefer FIRMS hotspots for place pointer; fall back to WFIGS incident
    place_src = nearest_firm or (wfigs_incident if (wfigs_corroborates and wfigs_incident) else None)
    if place_src is not None:
        approx_km = place_src.get("distance_km")
        if approx_km is None and isinstance(place_src.get("distance_miles"), (int, float)):
            approx_km = place_src.get("distance_miles") * 1.609344
    else:
        approx_km = None

    hypotheses.append({
        "id": "wildfire_smoke",
        "title": "Wildfire Smoke Transport",
        "score": smoke_score,
        "confidence": smoke_conf,
        "support": smoke_support,
        "against": smoke_against,
        "place": {
            "bearing": place_src.get("bearing") if place_src else None,
            "approx_km": approx_km,
            "description": place_desc or "Regional smoke transport"
        } if (nearest_firm or (incident_name and has_firms_corroboration) or (wfigs_corroborates and wfigs_incident)) else None
    })

    # Mode 2: Photochemical Ozone
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
    elif temp_f is not None and temp_f < p.ozone_cool_temp_f:
        o3_against.append(f"Cool ambient temperature ({temp_f}°F) suppresses rapid photochemical ozone generation")

    if o3_primary and is_hot:
        o3_conf = "high"
        o3_score = 85
    elif o3_primary:
        o3_conf = "medium"
        o3_score = 60
    elif is_hot and aqi_val is not None and aqi_val > p.aqi_elevated:
        o3_conf = "medium"
        o3_score = 45
    else:
        o3_conf = "low"
        o3_score = 15

    # Measured monitor ozone corroborates the ozone hypothesis
    if op_present and o3_primary and o3_ppb is not None and o3_ppb >= p.openaq_o3_ppb:
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

    # Mode 3: Windblown Dust 
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
    elif pm10_primary and wind_speed is not None and wind_speed >= p.dust_wind_speed_mph:
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

    # Mode 4: Winter Stagnation & Temperature Inversion
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
    if boundary_layer_height_m is not None and boundary_layer_height_m < p.shallow_boundary_layer_m:
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
        stagnation_conf = "high" if (boundary_layer_height_m is not None and boundary_layer_height_m < p.shallow_boundary_layer_m) else "medium"
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

    # Mode 5: Localized Urban and Industrial PM
    urban_support = []
    urban_against = []

    is_dust_or_stagnation = pm10_primary or (is_cold and is_calm)

    if pm_elevated and fire_signal_count == 0 and not has_haze and not is_dust_or_stagnation:
        urban_support.append(f"PM2.5 is elevated (AQI {aqi_val}) without verified fire, dust, or stagnation signals")
    elif pm_elevated and light_haze_extreme_pm and not is_dust_or_stagnation:
        urban_support.append(
            f"PM2.5 is very elevated (AQI {aqi_val}), but light haze suggests regional smoke may contribute alongside local sources"
        )
    elif pm_elevated and haze_no_fire and not is_dust_or_stagnation:
        urban_support.append(
            f"PM2.5 is elevated (AQI {aqi_val}) with regional haze but no verified fire evidence; local urban or regional sources remain a strong explanation"
        )
    elif pm_elevated:
        urban_support.append("PM is elevated, but a more specific mode (fire, dust, or stagnation) better explains it")

    if urban_tracer:
        parts = []
        if no2_ppb is not None and no2_ppb >= p.openaq_no2_ppb:
            parts.append("traffic-related nitrogen dioxide")
        if so2_ppb is not None and so2_ppb >= p.openaq_so2_ppb:
            parts.append("industrial sulfur dioxide")
        if co_ppm is not None and co_ppm >= p.openaq_co_ppm:
            parts.append("combustion-related carbon monoxide")
        urban_support.append(
            "Monitor data shows elevated " + " and ".join(parts) + ", a signature of local combustion and traffic"
        )
    if fine_dominated and pm_elevated and not (aod_present or has_smoke_corroboration):
        urban_support.append(
            f"PM is fine-particle dominated{monitor_distance_suffix}, consistent with local combustion or traffic rather than coarse dust"
        )

    if (
        op_present and pm_elevated and pm25_conc is not None
        and not measured_conflict and not (aod_present or has_smoke_corroboration)
    ):
        urban_support.append(measured_line)

    if not pm_elevated:
        urban_against.append("Surface PM is within satisfactory/normal range")
    if has_smoke_corroboration:
        urban_against.append(f"Verified fire evidence signals present ({fire_signal_count} indicator(s))")
    elif light_haze_extreme_pm:
        urban_against.append(
            "Light atmospheric haze with very unhealthy PM, favoring regional smoke settling over purely local urban emissions"
        )
    if is_dust_or_stagnation:
        urban_against.append("Conditions better match windblown dust or winter stagnation than generic urban/industrial PM")

    if pm_elevated and fire_signal_count == 0 and not has_haze and not is_dust_or_stagnation:
        # Clear AOD with elevated PM favors local urban emissions
        urban_conf, urban_score = "high", 75
    elif pm_elevated and light_haze_extreme_pm and not is_dust_or_stagnation:
        # Demote urban so soft smoke can rank above
        urban_conf, urban_score = "medium", 40
    elif pm_elevated and haze_no_fire and aod_value < p.aod_heavy and not is_dust_or_stagnation:
        # Medium haze without verified fire keeps urban competitive
        urban_conf, urban_score = "high", 70
    elif pm_elevated and haze_no_fire and aod_value >= p.aod_heavy and not is_dust_or_stagnation:
        # Heavy plume without verified fire competes but does not dominate
        urban_conf, urban_score = "medium", 50
    elif pm_elevated:
        urban_conf, urban_score = "medium", 35
    else:
        urban_conf, urban_score = "low", 15

    # Small rural communities lack local sources; urban stays capped at medium
    place = sig_map.get("place_context", {})
    rural_place = place.get("status") == "present" and place.get("rural") is True
    if (
        rural_place and pm_elevated
        and not has_smoke_corroboration and not is_dust_or_stagnation
    ):
        urban_conf, urban_score = "medium", 40
        open_questions.append(
            "This is a small community with limited local sources; elevated PM without verified fire, dust, or stagnation evidence may come from a source the available feeds do not capture."
        )

    if urban_tracer:
        urban_score = max(urban_score, 55)
        if urban_conf == "low":
            urban_conf = "medium"
    if fine_dominated and pm_elevated and not (aod_present or has_smoke_corroboration):
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

    if aqi_val is not None and aqi_val <= p.aqi_elevated:
        open_questions.append("Air quality index is currently in the 'Good' range (AQI ≤ 50).")
    if op_present and not measured_conflict and pm_elevated and daily_pct is not None and daily_pct >= p.daily_percentile_high:
        open_questions.append(
            "PM2.5 is well above this location's typical daily readings, an unusual day for this area"
        )
    # A distant low reading is spatial variation and will confuse users, only disclose nearby mismatches
    if (
        measured_conflict and conflict_value is not None
        and (openaq_dist_km is None or openaq_dist_km <= p.conflict_distance_km)
    ):
        category = observation.get("category", "elevated")
        dist_part = f" ({openaq_dist_km:.0f} km away)" if openaq_dist_km is not None else ""
        open_questions.append(
            f"The reported AQI is {category} ({aqi_val}), but the nearest air quality monitor "
            f"measures {conflict_value:.0f} micrograms per cubic meter right now{dist_part}; the AQI is "
            "a longer-term average while the monitor reading is current, so the two can differ"
        )
    return hypotheses, open_questions
