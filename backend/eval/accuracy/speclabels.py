"""Composition-based speciation labels — the first independent accuracy signal.

The rule-derived ``labels`` module re-applies the production thresholds to the
same archives the scorer consumes, so an evaluation against them measures
self-consistency. This module derives GROUND TRUTH from PM2.5 chemical
speciation (EPA AirData ``daily_SPEC_<year>``), using the published IMPROVE
soil / KNON formulas, and classifies each site-day into COMPOSITION classes
(``biomass_smoke`` / ``mineral_dust`` / ``mixed`` / ``secondary_aerosol`` /
``ambiguous``) from the chemistry alone. These labels are an offline answer
key only: they never touch the scorer or ``Params``.

IMPORTANT: these are COMPOSITION classes, not causal events. Chemistry says
what the PM is MADE OF, not which EVENT produced it. Fine-soil chemistry
(``mineral_dust``) can be windblown dust, road dust, construction, resuspended
soil, or transported mineral dust; elevated non-soil potassium
(``biomass_smoke``) indicates a biomass-combustion contribution, which includes
wildfire smoke but also wood heating and agricultural burning. The product's
event hypotheses (``wildfire_smoke``, ``windblown_dust``, ...) are validated
against these composition classes via the ``SPECIATION_TO_PRODUCT`` mapping in
``accuracy.__main__``.

Only IMPROVE-network rows are used (rows whose ``Method Name`` contains
"IMPROVE"); CSN is not harmonized. Formulas:

- Soil (µg/m³) = 2.53·Al + 2.86·Si + 1.87·Ca + 2.78·Fe + 2.23·Ti
  (IMPROVE SOP 351 / Malm et al. 1994). The file also carries a pre-computed
  ``Soil PM2.5 LC`` param (88348); prefer it when present.
- KNON (non-soil potassium, biomass-combustion tracer) = K − 0.6·Fe.
- Secondary aerosol = SO4 + NO3 (with NH4 and OC/EC tracked for the answer key).

Pure functions only — no I/O. Thresholds are module constants (LABEL
thresholds, deliberately independent of the scorer's ``Params``).
"""

from typing import Dict, Optional

# --- EPA AirData speciation parameter codes --------------------------------
PARAM_AL = "88104"
PARAM_SI = "88165"
PARAM_CA = "88111"
PARAM_FE = "88126"
PARAM_TI = "88161"
PARAM_SOIL_LC = "88348"  # pre-computed "Soil PM2.5 LC" (µg/m3)
PARAM_K = "88180"
PARAM_SO4 = "88403"
PARAM_NO3 = "88306"
PARAM_NH4 = "88301"
PARAM_OC = "88320"  # TOR
PARAM_EC = "88321"  # TOR

# --- IMPROVE soil-element regression coefficients (µg/m³ per µg/m³ element) -
# IMPROVE SOP 351 / Malm et al. 1994.
_SOIL_COEFFICIENTS = {
    PARAM_AL: 2.53,
    PARAM_SI: 2.86,
    PARAM_CA: 1.87,
    PARAM_FE: 2.78,
    PARAM_TI: 2.23,
}

# Ratio of potassium attributable to soil (K_soil ≈ 0.6·Fe), so the excess
# (KNON = K − 0.6·Fe) traces non-soil / smoke potassium. Malm et al. 1994.
KNON_SOIL_K_FE_RATIO = 0.6

# --- LABEL thresholds (transparent first pass) ------------------------------
# These are label-side constants, independent of the scorer's Params — the
# whole point of the speciation ground truth.
SMOKE_KNON_MIN = 0.1  # µg/m³ non-soil potassium for a biomass-combustion day
DUST_SOIL_MIN = 1.0  # µg/m³ IMPROVE soil for a mineral-dust-dominant day
# Soil threshold for a "mixed" day: soil present but below the dust-dominance
# threshold (DUST_SOIL_MIN) while the biomass signature is also elevated —
# neither a pure-smoke nor a pure-dust claim is fully correct for such a day.
MIXED_SOIL_MIN = 0.5  # µg/m³ IMPROVE soil to call a high-KNON day "mixed"
SECONDARY_MIN = 2.0  # µg/m³ SO4+NO3 for a secondary-aerosol day

# Composition classes (the speciation answer key). These describe what the PM
# is MADE OF, not the causal event — see the module docstring and the
# SPECIATION_TO_PRODUCT mapping in ``accuracy.__main__``.
SPECIATION_CLASSES = (
    "biomass_smoke",
    "mineral_dust",
    "mixed",
    "secondary_aerosol",
    "ambiguous",
)


def is_improve(method_name: Optional[str]) -> bool:
    """True when the row's ``Method Name`` is from the IMPROVE network.

    IMPROVE method names look like "IMPROVE Module A ... X-Ray Fluorescence"
    (elements) or "IMPROVE Module C ..." (carbon); CSN / Met One / other
    methods must not feed composition labels. None-safe.
    """
    if not method_name:
        return False
    return "IMPROVE" in method_name.upper()


def derive_components(
    spec_rows: Dict[str, float],
    method_name_by_param: Optional[Dict[str, str]] = None,
) -> Dict[str, Optional[float]]:
    """Derive IMPROVE composition components from one site-day's speciation.

    Args:
        spec_rows: ``parameter_code -> concentration`` for the day (the
            audit-complete row set from the store).
        method_name_by_param: ``parameter_code -> method_name``. When given,
            only IMPROVE-network rows contribute (``is_improve``); when None
            every row is treated as IMPROVE (permissive, for tests).

    Returns a dict with keys ``soil``, ``knon``, ``so4``, ``no3``, ``nh4``,
    ``oc``, ``ec`` — all Optional[float], None when the needed parameters are
    absent (or not IMPROVE).

    ``soil`` prefers the pre-computed ``Soil PM2.5 LC`` (88348) when present,
    else the Malm et al. 1994 element regression; ``knon = K − 0.6·Fe`` only
    when both K and Fe are present.
    """
    method_name_by_param = method_name_by_param or {}

    def _is_used(code: str) -> bool:
        if code not in spec_rows:
            return False
        if not method_name_by_param:
            # Permissive: no method info at all, treat every row as usable
            # (pure callers / tests that pass only concentrations).
            return True
        # With method info present, ONLY explicitly IMPROVE rows count — a row
        # without a stored method name cannot be verified as IMPROVE.
        method_name = method_name_by_param.get(code)
        return method_name is not None and is_improve(method_name)

    def _conc(code: str) -> Optional[float]:
        return spec_rows[code] if _is_used(code) else None

    soil_lc = _conc(PARAM_SOIL_LC)
    if soil_lc is not None:
        soil = soil_lc
    else:
        elements = {code: _conc(code) for code in _SOIL_COEFFICIENTS}
        if all(value is not None for value in elements.values()):
            # All five element concentrations are present (checked above), so
            # the regression can be evaluated.
            soil = sum(
                _SOIL_COEFFICIENTS[code] * elements[code]
                for code in _SOIL_COEFFICIENTS
            )
        else:
            soil = None

    k = _conc(PARAM_K)
    fe = _conc(PARAM_FE)
    knon = k - KNON_SOIL_K_FE_RATIO * fe if (k is not None and fe is not None) else None

    return {
        "soil": soil,
        "knon": knon,
        "so4": _conc(PARAM_SO4),
        "no3": _conc(PARAM_NO3),
        "nh4": _conc(PARAM_NH4),
        "oc": _conc(PARAM_OC),
        "ec": _conc(PARAM_EC),
    }


def classify_speciation(
    components: Dict[str, Optional[float]], elevated: bool = True
) -> str:
    """Classify one site-day into a COMPOSITION class.

    Args:
        components: ``derive_components`` output.
        elevated: whether the day is elevated (rule ``true_label`` not
            ``clean``/``ambiguous``). Non-elevated days carry no source
            attribution: a KNON spike on an AQI 30 day is not a smoke episode,
            so they classify ``ambiguous``. Defaults to True so a standalone
            chemistry-only classification works without the rule label.

    Thresholds are the module constants ``SMOKE_KNON_MIN``, ``DUST_SOIL_MIN``,
    ``MIXED_SOIL_MIN``, ``SECONDARY_MIN``. Precedence (first match wins):
    mixed, biomass_smoke, mineral_dust, secondary_aerosol, else ambiguous.

    The classes are composition classes, not causal events (see the module
    docstring): ``biomass_smoke`` means the PM carries a biomass-combustion
    contribution (wildfire smoke, wood heating, ag burning), and
    ``mineral_dust`` means fine-soil elements dominate (windblown dust, road
    dust, resuspended soil, transported dust). A day that is both high-KNON and
    high-soil is ``mixed`` (both signatures elevated); neither a pure-smoke nor
    a pure-dust product claim is fully correct for it.
    """
    if not elevated:
        return "ambiguous"

    knon = components.get("knon")
    soil = components.get("soil")
    so4 = components.get("so4")
    no3 = components.get("no3")

    smoke = knon is not None and knon >= SMOKE_KNON_MIN
    dust = soil is not None and soil >= DUST_SOIL_MIN

    # Both the biomass and mineral-dust signatures are elevated (soil present
    # at/above MIXED_SOIL_MIN, even below the dust-dominance threshold): a
    # mixed plume the first-pass chemistry cannot attribute to either.
    if smoke and soil is not None and soil >= MIXED_SOIL_MIN:
        return "mixed"

    if smoke and (soil is None or soil < DUST_SOIL_MIN):
        return "biomass_smoke"

    if dust and (knon is None or knon < SMOKE_KNON_MIN):
        return "mineral_dust"

    secondary = None
    if so4 is not None or no3 is not None:
        secondary = (so4 or 0.0) + (no3 or 0.0)
    if secondary is not None and secondary >= SECONDARY_MIN and (
        soil is None or secondary > soil
    ) and (knon is None or knon < SMOKE_KNON_MIN):
        return "secondary_aerosol"

    return "ambiguous"
