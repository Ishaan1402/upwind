"""Pure ground-truth label derivation for the accuracy-evaluation pipeline.

Labels are RULE-DERIVED: they reuse the same production thresholds and archived
evidence (AQS daily summaries, weather, HMS smoke plumes, FIRMS upwind fires)
that the scorer consumes, so an evaluation of the scorer against these labels
measures self-consistency, NOT independently-verified accuracy. See
``LABEL_KIND``. This module holds only the pure classification layer — no I/O.
Pulling records from the store per site-day and computing smoke/fire context is
the runner's job.

Classification precedence (deterministic, evaluated top to bottom):

  0. no determinable AQI (``observation.aqi is None``) -> ``ambiguous`` — a
     missing AQI is unknown, never "clean".
  1. ``aqi <= LABEL_PARAMS.aqi_elevated (50)`` -> ``clean`` — a non-elevated
     day is always clean, whatever the other signals say.
  2. PM2.5 primary:
       a. ``smoke_density in {"medium", "heavy"}`` -> ``wildfire_smoke``.
          A verified analyst plume TRUMPS inversion, so it wins even when the
          day is cold+calm (verified plume beats stagnation).
       b. ``upwind_fire is True`` -> ``wildfire_smoke``. This subsumes the
          explicit "light smoke + upwind fire" rule: a verified upwind fire
          makes even light analyst-observed smoke attributable to the fire.
       c. cold (``tmin_f < LABEL_PARAMS.cold_temp_f``) AND calm
          (``wind_max_mph < LABEL_PARAMS.calm_wind_speed_mph``):
            - light smoke WITHOUT an upwind fire -> ``ambiguous`` — light haze
              with no fire under cold/calm could be settling smoke or an
              inversion, so the light-haze exception takes precedence over
              winter stagnation.
            - otherwise -> ``winter_stagnation``.
       d. ``rural is True`` AND no upwind fire AND no smoke -> ``ambiguous`` —
          rural PM with no attributable source.
       e. otherwise -> ``urban_industrial_pm``.
  3. PM10 primary: ``wind_max_mph >= LABEL_PARAMS.dust_wind_speed_mph (8.0)`` ->
     ``windblown_dust``; else ``ambiguous`` (PM10 without wind is not clean
     dust).
  4. O3 primary -> ``ozone_episode``.
  5. NO2/CO/SO2 primary -> ``urban_industrial_pm`` (combustion tracers).
  6. no determinable primary -> ``ambiguous``.
"""

from typing import Dict, Iterable, List, Optional, Tuple

from backend.engine.params import LABEL_PARAMS
from backend.eval.accuracy.records import (
    AqsDailyRecord,
    LabelRecord,
    Observation,
    WeatherDailyRecord,
)

# Schema map: EPA AQS parameter code -> canonical pollutant name. This is a
# schema map (not a tunable threshold), so it lives here rather than in
# engine/params. ``88502`` is the non-FRM PM2.5 mass code IMPROVE speciation
# sites report their PM2.5 under, so it maps to PM2.5 like the FRM/FEM 88101.
PARAMETER_TO_POLLUTANT: Dict[str, str] = {
    "88101": "PM2.5",
    "88502": "PM2.5",
    "81102": "PM10",
    "44201": "O3",
    "42602": "NO2",
    "42101": "CO",
    "42401": "SO2",
}

# Canonical tie-break order for primary-pollutant selection (highest priority
# first): PM2.5 > PM10 > O3 > NO2 > SO2 > CO.
POLLUTANT_PRIORITY: Tuple[str, ...] = ("PM2.5", "PM10", "O3", "NO2", "SO2", "CO")
_POLLUTANT_RANK = {name: i for i, name in enumerate(POLLUTANT_PRIORITY)}

# Label classes: the five scorer hypothesis ids (engine/score.py) plus clean
# and ambiguous.
LABEL_CLASSES: Tuple[str, ...] = (
    "wildfire_smoke",
    "ozone_episode",
    "windblown_dust",
    "winter_stagnation",
    "urban_industrial_pm",
    "clean",
    "ambiguous",
)

# precision_tier is always "validated" while labels are AQS-backed.
PRECISION_TIER_VALIDATED = "validated"

# Labels are rule-derived: they re-apply the production thresholds to the same
# archives the scorer consumes. They are self-consistency checks, not
# independently-verified ground truth.
LABEL_KIND = "rule_derived"


def _aqi_display(aqi: Optional[int]) -> str:
    """AQI rendered into a reasoning string, None-safe ("unknown")."""
    return "unknown" if aqi is None else str(aqi)


def build_observation(aqs_records: Iterable[AqsDailyRecord]) -> Observation:
    """Aggregate a day's AQS parameter rows into a single Observation.

    ``aqi`` is the max non-null AQI across parameters; ``primary_pollutant`` is
    the pollutant carrying that max, with ties broken by canonical order
    (PM2.5 > PM10 > O3 > NO2 > SO2 > CO). Rows whose parameter code is not in
    ``PARAMETER_TO_POLLUTANT`` are ignored. If no row carries a non-null AQI the
    Observation is returned with ``aqi=None`` and an empty ``primary_pollutant``.
    """
    by_pollutant: Dict[str, List[int]] = {}
    concentrations: Dict[str, Optional[float]] = {}
    for rec in aqs_records:
        pollutant = PARAMETER_TO_POLLUTANT.get(rec.parameter_code)
        if pollutant is None:
            continue
        if rec.aqi is not None:
            by_pollutant.setdefault(pollutant, []).append(rec.aqi)
        if rec.concentration is not None:
            if pollutant not in concentrations or rec.concentration > concentrations[pollutant]:
                concentrations[pollutant] = rec.concentration

    pollutant_aqi = {name: max(aqis) for name, aqis in by_pollutant.items()}
    if not pollutant_aqi:
        return Observation(
            aqi=None,
            primary_pollutant="",
            pollutant_aqi={},
            concentrations=concentrations,
        )

    aqi = max(pollutant_aqi.values())
    # max over (aqi, -rank) picks the highest AQI and, on ties, the pollutant
    # highest in the canonical priority order.
    primary = max(
        pollutant_aqi,
        key=lambda p: (pollutant_aqi[p], -_POLLUTANT_RANK.get(p, len(_POLLUTANT_RANK))),
    )
    return Observation(
        aqi=aqi,
        primary_pollutant=primary,
        pollutant_aqi=pollutant_aqi,
        concentrations=concentrations,
    )


def classify_sample(
    observation: Observation,
    weather: Optional[WeatherDailyRecord],
    smoke_density: Optional[str] = None,
    upwind_fire: Optional[bool] = None,
    rural: Optional[bool] = None,
    *,
    site_id: Optional[str] = None,
    date_local: Optional[str] = None,
) -> LabelRecord:
    """Classify one site-day into a derived ground-truth label.

    Args:
        observation: The day's aggregated AQS state (see ``build_observation``).
        weather: The day's weather, or None (temp/wind then treated as unknown).
        smoke_density: None | "light" | "medium" | "heavy" (analyst HMS plume).
        upwind_fire: True/False/None (None = unknown) — an upwind fire source.
        rural: True/False/None — whether the site is in a rural area.
        site_id / date_local: Optional site-day identity for the LabelRecord.
            When omitted they fall back to the weather record's identity, and
            to "" when there is no weather record.

    Returns a ``LabelRecord`` whose ``label`` follows the module-level
    precedence and whose ``reasoning`` is a one-line description of the rule
    that fired.
    """
    if site_id is None:
        site_id = weather.site_id if weather is not None else ""
    if date_local is None:
        date_local = weather.date_local if weather is not None else ""

    cold = (
        weather is not None
        and weather.tmin_f is not None
        and weather.tmin_f < LABEL_PARAMS.cold_temp_f
    )
    calm = (
        weather is not None
        and weather.wind_max_mph is not None
        and weather.wind_max_mph < LABEL_PARAMS.calm_wind_speed_mph
    )
    aqi = observation.aqi
    primary = observation.primary_pollutant

    def _label(label: str, reasoning: str) -> LabelRecord:
        return LabelRecord(
            site_id=site_id,
            date_local=date_local,
            aqi=aqi,
            primary_pollutant=primary,
            label=label,
            precision_tier=PRECISION_TIER_VALIDATED,
            reasoning=reasoning,
        )

    # 0. No determinable AQI is unknown — never "clean".
    if aqi is None:
        return _label("ambiguous", "no determinable AQI")

    # 1. Non-elevated AQI is always clean, regardless of other signals.
    if aqi <= LABEL_PARAMS.aqi_elevated:
        return _label(
            "clean",
            f"AQI {_aqi_display(aqi)} at/below elevated threshold {LABEL_PARAMS.aqi_elevated}",
        )

    if primary == "PM2.5":
        # 2a. Verified analyst plume: medium/heavy smoke trumps inversion, so it
        #     wins even when cold+calm.
        if smoke_density in ("medium", "heavy"):
            return _label(
                "wildfire_smoke",
                f"PM2.5 primary with analyst-verified {smoke_density} smoke plume",
            )
        # 2b. Upwind fire present: smoke (including light smoke) is
        #     attributable to the fire.
        if upwind_fire is True:
            if smoke_density == "light":
                reason = "PM2.5 primary with light smoke and upwind fire present"
            else:
                reason = "PM2.5 primary with upwind fire present"
            return _label("wildfire_smoke", reason)
        # 2c. Cold + calm. The light-haze exception (light smoke without a fire
        #     under cold/calm could be settling smoke or an inversion) takes
        #     precedence over winter stagnation.
        if cold and calm:
            if smoke_density == "light" and upwind_fire is not True:
                return _label(
                    "ambiguous",
                    "PM2.5 primary, light haze without upwind fire under cold/calm "
                    "— settling smoke or inversion",
                )
            return _label(
                "winter_stagnation",
                f"PM2.5 primary, cold ({weather.tmin_f:.0f}F < {LABEL_PARAMS.cold_temp_f}) and "
                f"calm ({weather.wind_max_mph:.1f} mph < {LABEL_PARAMS.calm_wind_speed_mph})",
            )
        # 2d. Rural PM with no fire or smoke has no attributable source.
        if rural is True and upwind_fire is not True and smoke_density is None:
            return _label(
                "ambiguous",
                "PM2.5 primary at rural site with no fire or smoke attribution",
            )
        # 2e. Default: PM2.5 without fire/smoke/stagnation is urban/industrial.
        return _label(
            "urban_industrial_pm",
            "PM2.5 primary with no fire, smoke, or stagnation attribution",
        )

    if primary == "PM10":
        if (
            weather is not None
            and weather.wind_max_mph is not None
            and weather.wind_max_mph >= LABEL_PARAMS.dust_wind_speed_mph
        ):
            return _label(
                "windblown_dust",
                f"PM10 primary with wind {weather.wind_max_mph:.1f} mph "
                f"at/above {LABEL_PARAMS.dust_wind_speed_mph}",
            )
        # PM10 without dust-level wind is not clean dust.
        return _label(
            "ambiguous",
            "PM10 primary without dust-level wind",
        )

    if primary == "O3":
        return _label("ozone_episode", "O3 primary — ozone episode")

    if primary in ("NO2", "CO", "SO2"):
        return _label(
            "urban_industrial_pm",
            f"{primary} primary — urban/industrial combustion source",
        )

    # No determinable primary (e.g. an empty primary on a manually lifted AQI).
    return _label("ambiguous", "No determinable primary pollutant")
