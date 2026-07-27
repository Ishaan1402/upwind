from typing import Dict, Any, List, Tuple

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

    wind_speed = wind.get("speed_mph")
    boundary_layer_height_m = wind.get("boundary_layer_height_m")

    pm_primary = pm.get("primary", False)
    pm10_primary = pm.get("pm10_primary", False) or ("PM10" in primary)
    pm_elevated = pm.get("elevated", False) or (aqi_val > 50 and pm_primary)

    temp_f = o3.get("temperature_f")
    is_calm = wind_speed is not None and wind_speed < 4.0
    is_high_wind = wind_speed is not None and wind_speed >= 15.0
    is_cold = temp_f is not None and temp_f < 55.0

    # Count verified positive fire evidence indicators
    positive_fire_signals = []
    if aod_present and aod_value >= 0.4:
        positive_fire_signals.append(f"Dense atmospheric column particle plume detected (AOD {aod_value:.2f})")
    if firms_present and firms_count > 0:
        loc_str = f" ({nearest_firm['distance_miles']} mi {nearest_firm['bearing']})" if nearest_firm else ""
        positive_fire_signals.append(f"Detected {firms_count} active NASA FIRMS thermal hotspot cluster(s) upwind{loc_str}")
    if incident_name:
        positive_fire_signals.append(f"Confirmed active regional wildfire incident: '{incident_name}'")

    fire_signal_count = len(positive_fire_signals)

    # -------------------------------------------------------------
    # Mode 1: Wildfire Smoke Transport
    # -------------------------------------------------------------
    smoke_support = list(positive_fire_signals)
    smoke_against = []

    if pm_primary and pm_elevated:
        smoke_support.append(f"Surface PM2.5/PM10 is elevated as primary pollutant (AQI {aqi_val})")

    # Contradiction: Heavy column plume present but surface PM not elevated (Aloft smoke)
    if aod_value >= 0.4 and not pm_elevated:
        smoke_against.append(f"High atmospheric particle density overhead (AOD {aod_value:.2f}), but surface monitors show low ground PM levels")
        open_questions.append("Atmospheric particle plume is present overhead (aloft), but has not settled down to ground breathing level.")

    if firms.get("status") == "absent" and not aod_present and not incident_name:
        smoke_against.append("No active upwind fires or dense atmospheric plumes detected")

    # Deterministic Confidence Classification
    if aod_value >= 0.4 and not pm_elevated:
        smoke_conf = "low"
        smoke_score = 30
    elif fire_signal_count >= 2 and pm_elevated:
        smoke_conf = "high"
        smoke_score = 90
    elif fire_signal_count >= 1 and pm_elevated:
        smoke_conf = "high" if aod_value >= 0.8 else "medium"
        smoke_score = 75 if aod_value >= 0.8 else 65
    elif fire_signal_count >= 1:
        smoke_conf = "medium"
        smoke_score = 50
    else:
        smoke_conf = "low"
        smoke_score = 25

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
            "description": f"Incident: {incident_name}" if incident_name else (f"Upwind hotspots {nearest_firm['distance_miles']} mi {nearest_firm['bearing']}" if nearest_firm else "Regional smoke transport")
        } if (nearest_firm or incident_name) else None
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
        dust_against.append(f"Primary pollutant is {primary}, not coarse PM10 — dust is not the dominant particle fraction")

    if is_high_wind:
        dust_support.append(f"Sustained wind speed ({wind_speed} mph) is high enough to loft soil and dust")
    elif wind_speed is not None:
        dust_against.append(f"Wind speed ({wind_speed} mph) is below the threshold typically needed to loft significant dust")

    if pm10_primary and is_high_wind:
        dust_conf, dust_score = "high", 85
    elif pm10_primary and wind_speed is not None and wind_speed >= 8.0:
        dust_conf, dust_score = "medium", 55
    elif pm10_primary:
        dust_conf, dust_score = "low", 30
    else:
        dust_conf, dust_score = "low", 10

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
        stagnation_support.append("No upwind fire evidence — elevated PM is more likely local combustion/traffic trapped near the surface")
    else:
        stagnation_against.append("Active fire evidence present — elevated PM may be smoke-related rather than pure stagnation")
    if boundary_layer_height_m is not None and boundary_layer_height_m < 500:
        stagnation_support.append(f"Shallow boundary layer height ({boundary_layer_height_m:.0f}m) indicates trapped near-surface air")

    if pm_elevated and is_cold and is_calm and fire_signal_count == 0:
        stagnation_conf = "high" if (boundary_layer_height_m is not None and boundary_layer_height_m < 500) else "medium"
        stagnation_score = 85 if stagnation_conf == "high" else 65
    elif pm_elevated and is_cold and is_calm:
        stagnation_conf, stagnation_score = "low", 30
    else:
        stagnation_conf, stagnation_score = "low", 15

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
    urban_support = []
    urban_against = []

    is_dust_or_stagnation = pm10_primary or (is_cold and is_calm)

    if pm_elevated and fire_signal_count == 0 and not is_dust_or_stagnation:
        urban_support.append(f"PM2.5 is elevated (AQI {aqi_val}) without verified fire, dust, or stagnation signals")
    elif pm_elevated:
        urban_support.append("PM is elevated, but a more specific mode (fire, dust, or stagnation) better explains it")

    if not pm_elevated:
        urban_against.append("Surface PM is within satisfactory/normal range")
    if fire_signal_count > 0:
        urban_against.append(f"Verified fire evidence signals present ({fire_signal_count} indicator(s))")
    if is_dust_or_stagnation:
        urban_against.append("Conditions better match windblown dust or winter stagnation than generic urban/industrial PM")

    if pm_elevated and fire_signal_count == 0 and not is_dust_or_stagnation:
        urban_conf, urban_score = "high", 75
    elif pm_elevated:
        urban_conf, urban_score = "medium", 35
    else:
        urban_conf, urban_score = "low", 15

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

    return hypotheses, open_questions
