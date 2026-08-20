from unittest.mock import patch

from backend.services.geocode import geocode_location


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
