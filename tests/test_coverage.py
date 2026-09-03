from backend.services.coverage import coverage_for_location


def test_us_location():
    coverage = coverage_for_location({"country_code": "US"}, {"source": "AirNow"})
    assert coverage["mode"] == "us"
    assert coverage["country_code"] == "us"
    assert coverage["aqi_index"] == "US EPA AQI"


def test_international_location():
    coverage = coverage_for_location({"country_code": "CA"}, {"source": "Open-Meteo"})
    assert coverage["mode"] == "international"
    assert coverage["country_code"] == "ca"
    assert "not a local regulatory index" in coverage["disclaimer"]


def test_airnow_source_implies_us_when_country_unknown():
    coverage = coverage_for_location({"country_code": None}, {"source": "AirNow"})
    assert coverage["mode"] == "us"


def test_unknown_country():
    coverage = coverage_for_location({"country_code": None}, {"source": "Open-Meteo"})
    assert coverage["mode"] == "unknown"
