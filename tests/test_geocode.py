from unittest.mock import patch

from backend.services.geocode import geocode_location, _normalize_us_zip


def test_direct_latlon_includes_reverse_geocode_country():
    with patch(
        "backend.services.geocode._reverse_geocode",
        return_value={
            "state": "England",
            "city": "London",
            "country_code": "GB",
            "country": "United Kingdom",
        },
    ):
        result = geocode_location("51.5074,-0.1278")

    assert result["lat"] == 51.5074
    assert result["lon"] == -0.1278
    assert result["country_code"] == "GB"
    assert result["city"] == "London"


def test_direct_latlon_captures_us_zip_from_postcode():
    with patch(
        "backend.services.geocode._reverse_geocode",
        return_value={
            "state": "Oregon",
            "city": "Government Camp",
            "country_code": "US",
            "country": "United States",
            "postcode": "97028",
        },
    ):
        result = geocode_location("45.3040,-121.7540")

    assert result["zip_code"] == "97028"


def test_direct_latlon_strips_zip_plus_four():
    with patch(
        "backend.services.geocode._reverse_geocode",
        return_value={
            "state": "Oregon",
            "city": "Government Camp",
            "country_code": "us",
            "country": "United States",
            "postcode": "97028-1234",
        },
    ):
        result = geocode_location("45.3040,-121.7540")

    assert result["zip_code"] == "97028"


def test_direct_latlon_foreign_postcode_is_none():
    with patch(
        "backend.services.geocode._reverse_geocode",
        return_value={
            "state": "England",
            "city": "London",
            "country_code": "GB",
            "country": "United Kingdom",
            "postcode": "SW1A 1AA",
        },
    ):
        result = geocode_location("51.5074,-0.1278")

    assert result["zip_code"] is None


def test_normalize_us_zip_edge_cases():
    assert _normalize_us_zip("97028") == "97028"
    assert _normalize_us_zip("97028-1234") == "97028"
    assert _normalize_us_zip(" 97028 ") == "97028"
    assert _normalize_us_zip(None) is None
    assert _normalize_us_zip("") is None
    assert _normalize_us_zip("SW1A 1AA") is None
    assert _normalize_us_zip("1234") is None
