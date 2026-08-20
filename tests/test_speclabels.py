"""Tests for the speciation ground-truth label layer.

Covers the pure composition-label derivation (``speclabels``), the
``speciation`` store round-trip, and the streaming ingest parse. No network
access; speciation fixtures are hand-written CSV rows with the real EPA header.
"""

import csv

import pytest

from backend.eval.accuracy.ingest.speciation import (
    WESTERN_BBOX,
    parse_speciation_csv,
)
from backend.eval.accuracy.records import SpeciationRow
from backend.eval.accuracy.speclabels import (
    DUST_SOIL_MIN,
    KNON_SOIL_K_FE_RATIO,
    MIXED_SOIL_MIN,
    SECONDARY_MIN,
    SMOKE_KNON_MIN,
    classify_speciation,
    derive_components,
    is_improve,
)
from backend.eval.accuracy.store import AccuracyStore

# EPA speciation header, in published order (29 columns, verbatim).
_SPEC_FIELDS = [
    "State Code", "County Code", "Site Num", "Parameter Code", "POC",
    "Latitude", "Longitude", "Datum", "Parameter Name", "Sample Duration",
    "Pollutant Standard", "Date Local", "Units of Measure", "Event Type",
    "Observation Count", "Observation Percent", "Arithmetic Mean",
    "1st Max Value", "1st Max Hour", "AQI", "Method Code", "Method Name",
    "Local Site Name", "Address", "State Name", "County Name", "City Name",
    "CBSA Name", "Date of Last Change",
]

# IMPROVE + CSN method names used by the fixtures.
METHOD_IMPROVE_A = "IMPROVE Module A - Elements by X-Ray Fluorescence"
METHOD_IMPROVE_C = "IMPROVE Module C - Organic/Elemental Carbon by Thermal Optical Reflectance"
METHOD_MET_ONE = "Met One SASS - beta attenuation"

_ROW = {
    "State Code": "6", "County Code": "37", "Site Num": "1003",
    "POC": "1", "Latitude": "33.9372", "Longitude": "-118.1919",
    "Datum": "NAD83", "Sample Duration": "24 HOUR",
    "Pollutant Standard": "PM25 Speciation", "Date Local": "2020-07-01",
    "Units of Measure": "ug/m3 LC", "Event Type": "None",
    "Observation Count": "24", "Observation Percent": "100.0",
    "1st Max Value": "", "1st Max Hour": "", "AQI": "",
    "Local Site Name": "LA-North Main Street", "Address": "123 Main St",
    "State Name": "California", "County Name": "Los Angeles",
    "City Name": "Los Angeles", "CBSA Name": "Los Angeles-Long Beach-Anaheim, CA",
    "Date of Last Change": "2021-01-15",
}


def _spec_row(parameter_code, concentration, method_name, date_local="2020-07-01", **overrides):
    row = dict(_ROW)
    row.update({
        "Parameter Code": parameter_code,
        "Parameter Name": f"Param {parameter_code}",
        "Arithmetic Mean": str(concentration),
        "Method Code": "800" if "IMPROVE" in method_name else "145",
        "Method Name": method_name,
        "Date Local": date_local,
    })
    row.update(overrides)
    return row


def _build_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_ALL)
        writer.writerow(_SPEC_FIELDS)
        for row in rows:
            writer.writerow([row.get(field, "") for field in _SPEC_FIELDS])
    return path


def _components(rows):
    """Concentrations + method names from a list of row dicts, as the store
    returns them."""
    by_code = {}
    methods = {}
    for row in rows:
        code = row["Parameter Code"]
        by_code[code] = float(row["Arithmetic Mean"])
        if row["Method Name"]:
            methods[code] = row["Method Name"]
    return by_code, methods


# ---------------------------------------------------------------------------
# is_improve
# ---------------------------------------------------------------------------


def test_is_improve_matches_improve_network_names():
    assert is_improve(METHOD_IMPROVE_A) is True
    assert is_improve(METHOD_IMPROVE_C) is True
    # Case-insensitive.
    assert is_improve("improve module a - xrf") is True


def test_is_improve_rejects_non_improve_and_none():
    assert is_improve(METHOD_MET_ONE) is False
    assert is_improve("") is False
    assert is_improve(None) is False


# ---------------------------------------------------------------------------
# derive_components
# ---------------------------------------------------------------------------


def test_derive_components_soil_uses_malm_formula():
    rows = [
        _spec_row("88104", 0.10, METHOD_IMPROVE_A),  # Al
        _spec_row("88165", 0.50, METHOD_IMPROVE_A),  # Si
        _spec_row("88111", 0.20, METHOD_IMPROVE_A),  # Ca
        _spec_row("88126", 0.30, METHOD_IMPROVE_A),  # Fe
        _spec_row("88161", 0.05, METHOD_IMPROVE_A),  # Ti
    ]
    by_code, methods = _components(rows)
    components = derive_components(by_code, methods)

    # Soil = 2.53*Al + 2.86*Si + 1.87*Ca + 2.78*Fe + 2.23*Ti (Malm et al. 1994).
    expected = (
        2.53 * 0.10 + 2.86 * 0.50 + 1.87 * 0.20 + 2.78 * 0.30 + 2.23 * 0.05
    )
    assert components["soil"] == pytest.approx(expected)


def test_derive_components_soil_prefers_precomputed_88348():
    rows = [
        _spec_row("88348", 5.0, METHOD_IMPROVE_A),  # Soil PM2.5 LC
        _spec_row("88104", 0.10, METHOD_IMPROVE_A),
        _spec_row("88165", 0.50, METHOD_IMPROVE_A),
        _spec_row("88111", 0.20, METHOD_IMPROVE_A),
        _spec_row("88126", 0.30, METHOD_IMPROVE_A),
        _spec_row("88161", 0.05, METHOD_IMPROVE_A),
    ]
    by_code, methods = _components(rows)
    components = derive_components(by_code, methods)
    assert components["soil"] == pytest.approx(5.0)


def test_derive_components_soil_none_when_elements_incomplete():
    rows = [
        _spec_row("88104", 0.10, METHOD_IMPROVE_A),  # Al only: formula incomplete
    ]
    by_code, methods = _components(rows)
    components = derive_components(by_code, methods)
    assert components["soil"] is None


def test_derive_components_knon_is_k_minus_06_fe():
    rows = [
        _spec_row("88180", 1.0, METHOD_IMPROVE_A),  # K
        _spec_row("88126", 0.5, METHOD_IMPROVE_A),  # Fe
    ]
    by_code, methods = _components(rows)
    components = derive_components(by_code, methods)
    assert components["knon"] == pytest.approx(1.0 - KNON_SOIL_K_FE_RATIO * 0.5)


def test_derive_components_knon_none_when_k_or_fe_absent():
    only_k = [_spec_row("88180", 1.0, METHOD_IMPROVE_A)]
    by_code, methods = _components(only_k)
    assert derive_components(by_code, methods)["knon"] is None

    only_fe = [_spec_row("88126", 0.5, METHOD_IMPROVE_A)]
    by_code, methods = _components(only_fe)
    assert derive_components(by_code, methods)["knon"] is None


def test_derive_components_secondary_and_carbon_pass_through():
    rows = [
        _spec_row("88403", 1.5, METHOD_IMPROVE_A),  # SO4
        _spec_row("88306", 1.0, METHOD_IMPROVE_A),  # NO3
        _spec_row("88301", 0.8, METHOD_IMPROVE_A),  # NH4
        _spec_row("88320", 2.0, METHOD_IMPROVE_C),  # OC (TOR)
        _spec_row("88321", 0.4, METHOD_IMPROVE_C),  # EC (TOR)
    ]
    by_code, methods = _components(rows)
    components = derive_components(by_code, methods)
    assert components["so4"] == pytest.approx(1.5)
    assert components["no3"] == pytest.approx(1.0)
    assert components["nh4"] == pytest.approx(0.8)
    assert components["oc"] == pytest.approx(2.0)
    assert components["ec"] == pytest.approx(0.4)


def test_derive_components_ignores_non_improve_rows():
    # K measured by a CSN/SASS method must not feed KNON: both K and Fe are
    # then "absent" from the IMPROVE row set, so KNON is None.
    rows = [
        _spec_row("88180", 1.0, METHOD_MET_ONE),
        _spec_row("88126", 0.5, METHOD_IMPROVE_A),
    ]
    by_code, methods = _components(rows)
    components = derive_components(by_code, methods)
    assert components["knon"] is None


def test_derive_components_excludes_rows_without_verifiable_method():
    # When method info is provided, a param with no stored method name cannot
    # be verified as IMPROVE and must not feed the components.
    by_code = {"88180": 1.0, "88126": 0.5}
    methods = {"88126": METHOD_IMPROVE_A}  # Fe verified IMPROVE, K has none
    components = derive_components(by_code, methods)
    assert components["knon"] is None


# ---------------------------------------------------------------------------
# classify_speciation
# ---------------------------------------------------------------------------


def _all_none():
    return {
        "soil": None, "knon": None, "so4": None, "no3": None,
        "nh4": None, "oc": None, "ec": None,
    }


def test_classify_biomass_smoke_when_knon_high_and_soil_low():
    components = _all_none()
    components.update({"knon": SMOKE_KNON_MIN + 0.4, "soil": 0.2})
    assert classify_speciation(components) == "biomass_smoke"
    # Smoke with no soil measurement at all also classifies biomass_smoke.
    components.update({"soil": None})
    assert classify_speciation(components) == "biomass_smoke"


def test_classify_mineral_dust_when_soil_high_and_knon_low():
    components = _all_none()
    components.update({"soil": DUST_SOIL_MIN + 0.5, "knon": 0.05})
    assert classify_speciation(components) == "mineral_dust"


def test_classify_mixed_when_both_signals_elevated():
    # Both the biomass and mineral signatures are high (knon at/above
    # SMOKE_KNON_MIN and soil at/above MIXED_SOIL_MIN): neither a pure-smoke
    # nor a pure-dust claim is fully correct.
    components = _all_none()
    components.update({"knon": SMOKE_KNON_MIN + 0.4, "soil": 2.0})
    assert classify_speciation(components) == "mixed"


def test_classify_mixed_when_soil_present_below_dust_dominance():
    # Soil above MIXED_SOIL_MIN but below DUST_SOIL_MIN while biomass is also
    # elevated is still "mixed" (soil present but below the dust-dominance
    # threshold) — previously this classified as wildfire_smoke.
    components = _all_none()
    components.update({"knon": SMOKE_KNON_MIN + 0.4, "soil": MIXED_SOIL_MIN + 0.1})
    assert MIXED_SOIL_MIN < DUST_SOIL_MIN
    assert classify_speciation(components) == "mixed"
    # Below MIXED_SOIL_MIN the soil signal is too weak to count: biomass_smoke.
    components.update({"soil": MIXED_SOIL_MIN - 0.1})
    assert classify_speciation(components) == "biomass_smoke"


def test_classify_secondary_when_sulfate_nitrate_high():
    components = _all_none()
    components.update({"so4": 1.5, "no3": 1.0})
    # 1.5 + 1.0 = 2.5 >= SECONDARY_MIN, no soil/knon to override.
    assert SECONDARY_MIN == 2.0
    assert classify_speciation(components) == "secondary_aerosol"


def test_classify_ambiguous_when_all_low_or_absent():
    assert classify_speciation(_all_none()) == "ambiguous"
    low = _all_none()
    low.update({"knon": 0.05, "soil": 0.3, "so4": 0.5, "no3": 0.5})
    assert classify_speciation(low) == "ambiguous"


def test_classify_non_elevated_day_is_ambiguous():
    components = _all_none()
    components.update({"knon": 0.5, "soil": 0.2})
    # Chemistry on a non-elevated day carries no source attribution.
    assert classify_speciation(components, elevated=False) == "ambiguous"


# ---------------------------------------------------------------------------
# store round-trip + join
# ---------------------------------------------------------------------------


def _spec_records():
    return [
        SpeciationRow(
            site_id="06-037-1003", date_local="2020-07-01",
            parameter_code="88104", parameter_name="Aluminum",
            method_code="800", method_name=METHOD_IMPROVE_A,
            concentration=0.10, units="ug/m3 LC",
        ),
        SpeciationRow(
            site_id="06-037-1003", date_local="2020-07-01",
            parameter_code="88180", parameter_name="Potassium",
            method_code="800", method_name=METHOD_IMPROVE_A,
            concentration=1.0, units="ug/m3 LC",
        ),
        SpeciationRow(
            site_id="06-037-1003", date_local="2020-07-02",
            parameter_code="88180", parameter_name="Potassium",
            method_code="800", method_name=METHOD_IMPROVE_A,
            concentration=2.0, units="ug/m3 LC",
        ),
    ]


def test_speciation_store_roundtrip_and_join(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        records = _spec_records()
        assert store.insert_speciation(records) == 3
        assert store.count_speciation() == 3

        concs = store.fetch_speciation("06-037-1003", "2020-07-01")
        assert concs == {"88104": 0.10, "88180": 1.0}
        methods = store.fetch_speciation_methods("06-037-1003", "2020-07-01")
        assert methods == {"88104": METHOD_IMPROVE_A, "88180": METHOD_IMPROVE_A}

        # Idempotent: re-inserting replaces in place, no duplicates.
        assert store.insert_speciation(records) == 3
        assert store.count_speciation() == 3

        # Nothing to join against until predictions exist.
        assert store.fetch_speciation_join_site_days() == []

        # A prediction on the same site-day joins; a non-overlapping one does not.
        from backend.eval.accuracy.records import PredictionRecord

        store.insert_predictions([
            PredictionRecord(
                site_id="06-037-1003", date_local="2020-07-01",
                true_label="wildfire_smoke", predicted_label="wildfire_smoke",
                top_score=0.9, top_confidence="high",
            ),
            PredictionRecord(
                site_id="06-037-1003", date_local="2020-07-03",
                true_label="clean", predicted_label="clean",
                top_score=0.1, top_confidence="low",
            ),
        ])
        assert store.fetch_speciation_join_site_days() == [
            ("06-037-1003", "2020-07-01"),
        ]
    finally:
        store.close()


# ---------------------------------------------------------------------------
# ingest parse (streaming, bbox-filtered)
# ---------------------------------------------------------------------------


def test_parse_speciation_csv_streams_and_filters_bbox(tmp_path):
    rows = [
        # Inside the Western bbox (CA).
        _spec_row("88104", 0.10, METHOD_IMPROVE_A),
        # Inside the bbox but non-numeric concentration -> skipped.
        _spec_row("88165", "", METHOD_IMPROVE_A),
        # Outside the bbox (longitude -90, east of -102) -> skipped.
        _spec_row("88104", 0.20, METHOD_IMPROVE_A, **{"Longitude": "-90.0"}),
        # North of the bbox (lat 52) -> skipped.
        _spec_row("88104", 0.30, METHOD_IMPROVE_A, **{"Latitude": "52.0"}),
    ]
    csv_path = _build_csv(rows, tmp_path / "daily_SPEC_2020.csv")

    parsed = parse_speciation_csv(csv_path, bbox=WESTERN_BBOX)
    assert len(parsed) == 1
    rec = parsed[0]
    assert isinstance(rec, SpeciationRow)
    # site_id zero-padded exactly as the AQS adapter builds it.
    assert rec.site_id == "06-037-1003"
    assert rec.date_local == "2020-07-01"
    assert rec.parameter_code == "88104"
    assert rec.concentration == pytest.approx(0.10)
    assert rec.method_name == METHOD_IMPROVE_A
    assert rec.units == "ug/m3 LC"
    # The SPEC CSV's coordinates ride along on the row (for the site table).
    assert rec.lat == pytest.approx(33.9372)
    assert rec.lon == pytest.approx(-118.1919)


# ---------------------------------------------------------------------------
# speciation_sites store (site coordinates for the weather backfill)
# ---------------------------------------------------------------------------


def test_speciation_sites_store_roundtrip_and_idempotent(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        assert store.fetch_speciation_sites() == []

        sites = [
            ("06-037-1003", 33.9372, -118.1919),
            ("35-013-0002", 36.2664, -115.2201),
        ]
        assert store.insert_speciation_sites(sites) == 2
        assert store.fetch_speciation_sites() == [
            ("06-037-1003", 33.9372, -118.1919),
            ("35-013-0002", 36.2664, -115.2201),
        ]

        # INSERT OR REPLACE under the site_id primary key is idempotent.
        assert store.insert_speciation_sites(sites) == 2
        assert len(store.fetch_speciation_sites()) == 2

        # Re-ingesting with a refreshed coordinate replaces in place.
        store.insert_speciation_sites([("06-037-1003", 33.94, -118.19)])
        by_id = {site_id: (lat, lon) for site_id, lat, lon in store.fetch_speciation_sites()}
        assert by_id["06-037-1003"] == (33.94, -118.19)
        assert len(store.fetch_speciation_sites()) == 2
    finally:
        store.close()
