"""Tests for the METAR dust present-weather confirmation (backend/services/metar.py)."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

from backend.services.metar import (
    fetch_metar_dust,
    match_dust_phenomenon,
    DUST_PHENOMENA,
    METAR_BBOX_DEG,
)


def _metar(icao, raw, wx=None):
    obs = {
        "icaoId": icao,
        "name": f"{icao} Muni, AZ, US",
        "lat": 33.4,
        "lon": -111.9,
        "rawOb": raw,
    }
    if wx is not None:
        obs["wxString"] = wx
    return obs


def _mock_client(payload=None, status=200, raise_exc=False):
    mock_client = AsyncMock()
    if raise_exc:
        mock_client.get.side_effect = RuntimeError("network down")
    else:
        response = Mock()
        response.status_code = status
        response.json.return_value = payload
        mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


def test_dust_phenomena_are_standard_codes():
    assert DUST_PHENOMENA == ("BLDU", "DU", "DS", "SS", "VCDS", "PO", "TSDS")


def test_match_dust_phenomenon_requires_standalone_token():
    # The code must be a standalone token: "DU" inside "DULUTH" or "BLDU"
    # must not false-positive.
    assert match_dust_phenomenon("METAR KPHX 201151Z 24025KT 10SM CLR", None) is None
    assert match_dust_phenomenon("METAR KDUL 201151Z 24025KT 10SM BLDU", None) == "BLDU"
    assert match_dust_phenomenon("METAR KDUL 201151Z 24025KT 3SM DU", None) == "DU"
    assert match_dust_phenomenon("METAR KDUL 201151Z 24025KT 3SM +BLDU", None) == "BLDU"


def test_match_dust_phenomenon_handles_each_code():
    for code in DUST_PHENOMENA:
        assert match_dust_phenomenon(f"METAR KDUL 201151Z 24025KT 2SM {code}", None) == code


def test_match_dust_phenomenon_checks_wx_string():
    assert match_dust_phenomenon("METAR KDUL 201151Z 24025KT 10SM CLR", "VCDS") == "VCDS"
    assert match_dust_phenomenon("", "TSDS") == "TSDS"
    assert match_dust_phenomenon(None, None) is None
    assert match_dust_phenomenon("", "") is None


def test_fetch_metar_dust_present_returns_station_and_phenomenon():
    payload = [
        _metar("KPHX", "METAR KPHX 201151Z 24010KT 10SM CLR 33/10 A3001"),
        _metar(
            "KDUL",
            "METAR KDUL 201151Z 24025G35KT 2SM BLDU HZ 31/08 A2995",
            wx="BLDU HZ",
        ),
    ]
    with patch("backend.services.metar.httpx.AsyncClient", return_value=_mock_client(payload)):
        res = asyncio.run(fetch_metar_dust(31.76, -106.49))

    assert res["status"] == "present"
    assert res["station"] == "KDUL"
    assert res["phenomenon"] == "BLDU"
    assert "BLDU" in res["raw"]


def test_fetch_metar_dust_present_from_wx_string_without_raw_code():
    payload = [
        _metar("KELP", "METAR KELP 201151Z 24025KT 10SM CLR 33/10 A3001", wx=""),
        _metar("KDNA", "METAR KDNA 201151Z 31030KT 6SM HZ 30/09 A3009", wx="VCDS"),
    ]
    with patch("backend.services.metar.httpx.AsyncClient", return_value=_mock_client(payload)):
        res = asyncio.run(fetch_metar_dust(31.76, -106.49))

    assert res["status"] == "present"
    assert res["station"] == "KDNA"
    assert res["phenomenon"] == "VCDS"


def test_fetch_metar_dust_absent_when_no_dust_codes():
    payload = [
        _metar("KELP", "METAR KELP 201151Z 24010KT 10SM CLR 33/10 A3001", wx=""),
        _metar("KDNA", "METAR KDNA 201151Z 31008KT 10SM CLR 30/09 A3009"),
    ]
    with patch("backend.services.metar.httpx.AsyncClient", return_value=_mock_client(payload)):
        res = asyncio.run(fetch_metar_dust(31.76, -106.49))

    assert res["status"] == "absent"
    assert res["station"] is None
    assert res["phenomenon"] is None


def test_fetch_metar_dust_absent_on_empty_list():
    with patch("backend.services.metar.httpx.AsyncClient", return_value=_mock_client([])):
        res = asyncio.run(fetch_metar_dust(31.76, -106.49))
    assert res["status"] == "absent"
    assert res["phenomenon"] is None


def test_fetch_metar_dust_unavailable_on_http_error():
    with patch("backend.services.metar.httpx.AsyncClient", return_value=_mock_client(status=503)):
        res = asyncio.run(fetch_metar_dust(31.76, -106.49))
    assert res["status"] == "unavailable"
    assert "HTTP 503" in res["details"]


def test_fetch_metar_dust_unavailable_on_exception():
    with patch("backend.services.metar.httpx.AsyncClient", return_value=_mock_client(raise_exc=True)):
        res = asyncio.run(fetch_metar_dust(31.76, -106.49))
    assert res["status"] == "unavailable"
    assert res["station"] is None
    assert res["phenomenon"] is None
    assert "details" in res


def test_fetch_metar_dust_unavailable_on_unexpected_payload():
    with patch("backend.services.metar.httpx.AsyncClient", return_value=_mock_client({"error": "boom"})):
        res = asyncio.run(fetch_metar_dust(31.76, -106.49))
    assert res["status"] == "unavailable"


def test_fetch_metar_dust_builds_bbox_around_target():
    mock_client = _mock_client([])
    with patch("backend.services.metar.httpx.AsyncClient", return_value=mock_client):
        asyncio.run(fetch_metar_dust(31.76, -106.49))

    params = mock_client.get.call_args.kwargs["params"]
    assert params["format"] == "json"
    expected = (
        f"{31.76 - METAR_BBOX_DEG},{-106.49 - METAR_BBOX_DEG},"
        f"{31.76 + METAR_BBOX_DEG},{-106.49 + METAR_BBOX_DEG}"
    )
    assert params["bbox"] == expected
