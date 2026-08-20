"""Known-label validation set for the LLM judge.

The current set is intentionally mechanical: for each scenario we construct a
grounded narrative from the deterministic fallback generator and an
adversarial narrative with a planted hallucination. That gives us an objective
pass/fail label for the judge to be measured against. Treat these labels as a
v1 scaffold and replace/extend them with human-reviewed cases as the project
grows.

Usage:
    python -m backend.eval_validation --out validation_results.json
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.eval_corpus import CORPUS
from backend.engine.score import score_hypotheses
from backend.llm import generate_fallback_narrative
from backend.llm_judge import judge_narrative


INTERNATIONAL_SCENARIOS: List[Dict[str, Any]] = [
    {
        "name": "vancouver_wildfire_smoke",
        "location": {"lat": 49.2827, "lon": -123.1207, "name": "Vancouver", "zip_code": None, "state": "BC", "city": "Vancouver", "country_code": "CA", "country": "Canada"},
        "observation": {"aqi": 132, "primary_pollutant": "PM2.5", "category": "Unhealthy for Sensitive Groups"},
        "signals": [
            {"id": "aerosol_plume", "status": "present", "density": "medium", "aod_value": 0.62},
            {"id": "firms_upwind", "status": "present", "count": 4, "nearest": {"distance_miles": 38.0, "bearing": "NE", "distance_km": 61.0}, "incident_name": "Cameron Lake Fire"},
            {"id": "wind", "status": "present", "speed_mph": 9.0, "direction_deg": 225.0, "boundary_layer_height_m": 1100.0},
            {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
            {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 68.0},
            {"id": "openaq_concentrations", "status": "present", "pm25": 31.0, "pm10": 38.0, "pm25_pm10_ratio": 0.82, "monitor": {"name": "Vancouver", "distance_km": 6.0, "provider": "BC Air Quality"}},
        ],
    },
    {
        "name": "london_urban_pm",
        "location": {"lat": 51.5074, "lon": -0.1278, "name": "London", "zip_code": None, "state": None, "city": "London", "country_code": "GB", "country": "United Kingdom"},
        "observation": {"aqi": 88, "primary_pollutant": "PM2.5", "category": "Moderate"},
        "signals": [
            {"id": "aerosol_plume", "status": "absent", "aod_value": 0.11},
            {"id": "firms_upwind", "status": "absent", "count": 0},
            {"id": "wind", "status": "present", "speed_mph": 7.0, "direction_deg": 240.0, "boundary_layer_height_m": 700.0},
            {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
            {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 62.0},
            {"id": "openaq_concentrations", "status": "present", "pm25": 22.0, "pm10": 26.0, "pm25_pm10_ratio": 0.85, "no2_ppb": 58.0, "monitor": {"name": "Marylebone", "distance_km": 3.0, "provider": "UK AURN"}},
        ],
    },
    {
        "name": "delhi_windblown_dust",
        "location": {"lat": 28.6139, "lon": 77.2090, "name": "New Delhi", "zip_code": None, "state": "DL", "city": "New Delhi", "country_code": "IN", "country": "India"},
        "observation": {"aqi": 178, "primary_pollutant": "PM10", "category": "Unhealthy"},
        "signals": [
            {"id": "aerosol_plume", "status": "absent", "aod_value": 0.18},
            {"id": "firms_upwind", "status": "absent", "count": 0},
            {"id": "wind", "status": "present", "speed_mph": 24.0, "direction_deg": 290.0, "boundary_layer_height_m": 1600.0},
            {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": True, "pm25_primary": False, "elevated": True},
            {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 88.0},
            {"id": "openaq_concentrations", "status": "present", "pm25": 42.0, "pm10": 205.0, "pm25_pm10_ratio": 0.20, "monitor": {"name": "ITO", "distance_km": 4.0, "provider": "CPCB"}},
        ],
    },
    {
        "name": "sydney_ozone_episode",
        "location": {"lat": -33.8688, "lon": 151.2093, "name": "Sydney", "zip_code": None, "state": "NSW", "city": "Sydney", "country_code": "AU", "country": "Australia"},
        "observation": {"aqi": 118, "primary_pollutant": "O3", "category": "Unhealthy for Sensitive Groups"},
        "signals": [
            {"id": "aerosol_plume", "status": "absent", "aod_value": 0.09},
            {"id": "firms_upwind", "status": "absent", "count": 0},
            {"id": "wind", "status": "present", "speed_mph": 5.0, "direction_deg": 40.0, "boundary_layer_height_m": 1300.0},
            {"id": "surface_pm_level", "status": "absent", "primary": False, "pm10_primary": False, "pm25_primary": False, "elevated": False},
            {"id": "ozone_heat", "status": "present", "primary": True, "hot_day": True, "temperature_f": 94.0},
        ],
    },
    {
        "name": "toronto_winter_stagnation",
        "location": {"lat": 43.6532, "lon": -79.3832, "name": "Toronto", "zip_code": None, "state": "ON", "city": "Toronto", "country_code": "CA", "country": "Canada"},
        "observation": {"aqi": 115, "primary_pollutant": "PM2.5", "category": "Unhealthy for Sensitive Groups"},
        "signals": [
            {"id": "aerosol_plume", "status": "absent", "aod_value": 0.12},
            {"id": "firms_upwind", "status": "absent", "count": 0},
            {"id": "wind", "status": "present", "speed_mph": 3.0, "direction_deg": 10.0, "boundary_layer_height_m": 320.0},
            {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
            {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 22.0},
            {"id": "openaq_concentrations", "status": "present", "pm25": 29.0, "pm10": 34.0, "same_hour_percentile": 96.0, "same_hour_median": 9.0, "monitor": {"name": "Downtown Toronto", "distance_km": 5.0, "provider": "Ontario MOE"}},
        ],
    },
    {
        "name": "mexico_city_urban_pm",
        "location": {"lat": 19.4326, "lon": -99.1332, "name": "Mexico City", "zip_code": None, "state": "CDMX", "city": "Mexico City", "country_code": "MX", "country": "Mexico"},
        "observation": {"aqi": 95, "primary_pollutant": "PM2.5", "category": "Moderate"},
        "signals": [
            {"id": "aerosol_plume", "status": "absent", "aod_value": 0.14},
            {"id": "firms_upwind", "status": "absent", "count": 0},
            {"id": "wind", "status": "present", "speed_mph": 6.0, "direction_deg": 160.0, "boundary_layer_height_m": 800.0},
            {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
            {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 71.0},
            {"id": "openaq_concentrations", "status": "present", "pm25": 24.0, "pm10": 28.0, "pm25_pm10_ratio": 0.86, "no2_ppb": 54.0, "monitor": {"name": "Merced", "distance_km": 2.0, "provider": "SEDEMA"}},
        ],
    },
    {
        "name": "sao_paulo_ozone",
        "location": {"lat": -23.5505, "lon": -46.6333, "name": "São Paulo", "zip_code": None, "state": "SP", "city": "São Paulo", "country_code": "BR", "country": "Brazil"},
        "observation": {"aqi": 122, "primary_pollutant": "O3", "category": "Unhealthy for Sensitive Groups"},
        "signals": [
            {"id": "aerosol_plume", "status": "absent", "aod_value": 0.10},
            {"id": "firms_upwind", "status": "absent", "count": 0},
            {"id": "wind", "status": "present", "speed_mph": 6.0, "direction_deg": 120.0, "boundary_layer_height_m": 1400.0},
            {"id": "surface_pm_level", "status": "absent", "primary": False, "pm10_primary": False, "pm25_primary": False, "elevated": False},
            {"id": "ozone_heat", "status": "present", "primary": True, "hot_day": True, "temperature_f": 89.0},
        ],
    },
    {
        "name": "tokyo_winter_stagnation",
        "location": {"lat": 35.6762, "lon": 139.6503, "name": "Tokyo", "zip_code": None, "state": None, "city": "Tokyo", "country_code": "JP", "country": "Japan"},
        "observation": {"aqi": 108, "primary_pollutant": "PM2.5", "category": "Unhealthy for Sensitive Groups"},
        "signals": [
            {"id": "aerosol_plume", "status": "absent", "aod_value": 0.11},
            {"id": "firms_upwind", "status": "absent", "count": 0},
            {"id": "wind", "status": "present", "speed_mph": 2.0, "direction_deg": 350.0, "boundary_layer_height_m": 300.0},
            {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
            {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 38.0},
            {"id": "openaq_concentrations", "status": "present", "pm25": 27.0, "pm10": 31.0, "same_hour_percentile": 93.0, "same_hour_median": 11.0, "monitor": {"name": "Ichigaya", "distance_km": 4.0, "provider": "Tokyo Metropolitan"}},
        ],
    },
]


VALIDATION_SCENARIOS: List[Dict[str, Any]] = CORPUS + INTERNATIONAL_SCENARIOS


def build_validation_set() -> List[Dict[str, Any]]:
    """Build 2 cases per scenario: a grounded pass and a hallucinated fail."""
    cases: List[Dict[str, Any]] = []
    for scenario in VALIDATION_SCENARIOS:
        location = scenario["location"]
        observation = scenario["observation"]
        signals = scenario["signals"]
        hypotheses, open_questions = score_hypotheses(observation, signals)
        evidence = {
            "location": location,
            "observation": observation,
            "signals": signals,
            "hypotheses": hypotheses,
            "open_questions": open_questions,
        }
        grounded = generate_fallback_narrative(
            location, observation, signals, hypotheses, open_questions
        )
        hallucinated = (
            grounded
            + "\n\nA wildfire named Nonexistent Fire is actively burning five miles from here "
            + "and is the primary cause of the current conditions."
        )
        cases.append({
            "name": f"{scenario['name']}_grounded",
            "evidence": evidence,
            "narrative": grounded,
            "gold_verdict": "pass",
            "label_method": "deterministic-construction v1",
            "notes": "Generated by the deterministic fallback; should be fully grounded.",
        })
        cases.append({
            "name": f"{scenario['name']}_hallucinated",
            "evidence": evidence,
            "narrative": hallucinated,
            "gold_verdict": "fail",
            "label_method": "deterministic-construction v1",
            "notes": "Planted unsupported wildfire claim; judge should fail it.",
        })
    return cases


def _cohens_kappa(a: List[str], b: List[str]) -> Optional[float]:
    """Binary Cohen's kappa for pass/fail ratings, ignoring skipped/unknown."""
    pairs = [(x, y) for x, y in zip(a, b) if x in ("pass", "fail") and y in ("pass", "fail")]
    if not pairs:
        return None
    n = len(pairs)
    a_pass = sum(1 for x, _ in pairs if x == "pass")
    b_pass = sum(1 for _, y in pairs if y == "pass")
    agree = sum(1 for x, y in pairs if x == y)
    po = agree / n
    pe = (a_pass / n) * (b_pass / n) + ((n - a_pass) / n) * ((n - b_pass) / n)
    return round((po - pe) / (1 - pe), 4) if pe < 1 else None


async def run_validation() -> Dict[str, Any]:
    cases = build_validation_set()
    results: List[Dict[str, Any]] = []
    gold: List[str] = []
    predicted: List[str] = []
    for case in cases:
        verdict = await judge_narrative(case["evidence"], case["narrative"])
        judge_value = verdict.get("verdict")
        gold_value = case["gold_verdict"]
        agreement = (
            judge_value == gold_value
            if judge_value in ("pass", "fail")
            else None
        )
        gold.append(gold_value)
        predicted.append(judge_value or "unknown")
        results.append({
            "name": case["name"],
            "gold_verdict": gold_value,
            "judge_verdict": judge_value,
            "agreement": agreement,
            "judge_model": verdict.get("judge_model"),
            "reasoning": verdict.get("reasoning"),
        })

    judged_pairs = [
        (g, p) for g, p in zip(gold, predicted) if p in ("pass", "fail")
    ]
    exact_agreement = (
        round(sum(1 for g, p in judged_pairs if g == p) / len(judged_pairs), 4)
        if judged_pairs
        else None
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_cases": len(cases),
        "judged_cases": len(judged_pairs),
        "exact_agreement": exact_agreement,
        "cohens_kappa": _cohens_kappa(gold, predicted),
        "confusion": {
            "gold_pass_judge_pass": sum(1 for g, p in judged_pairs if g == "pass" and p == "pass"),
            "gold_pass_judge_fail": sum(1 for g, p in judged_pairs if g == "pass" and p == "fail"),
            "gold_fail_judge_pass": sum(1 for g, p in judged_pairs if g == "fail" and p == "pass"),
            "gold_fail_judge_fail": sum(1 for g, p in judged_pairs if g == "fail" and p == "fail"),
        },
        "results": results,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="backend.eval_validation", description=__doc__)
    parser.add_argument("--out", help="optional JSON output path")
    args = parser.parse_args(argv)
    result = asyncio.run(run_validation())
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote validation results to {args.out}")
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
