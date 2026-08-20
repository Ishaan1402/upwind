import asyncio
from unittest.mock import AsyncMock, Mock, patch
from backend.services import place_context
from backend.services.place_context import fetch_place_context, _parse_population


def test_parse_population_ok():
    payload = [["NAME", "P1_001N", "zip code tabulation area"], ["ZCTA5 97028", "230", "97028"]]
    assert _parse_population(payload) == 230


def test_parse_population_malformed():
    assert _parse_population([]) is None
    assert _parse_population([["NAME", "P1_001N", "zip code tabulation area"]]) is None
    assert _parse_population("nope") is None
    assert _parse_population([["NAME", "P1_001N", "x"], ["ZCTA5 97028", "not-a-number", "97028"]]) is None


def _mock_client(payload=None, raise_exc=False, status=200):
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


def test_fetch_place_context_rural_small_pop():
    payload = [["NAME", "P1_001N", "zip code tabulation area"], ["ZCTA5 97028", "230", "97028"]]
    with patch.object(place_context, "CENSUS_API_KEY", "test-key"), \
         patch.object(place_context, "get_cached_place", return_value=None), \
         patch.object(place_context, "set_cached_place") as set_cache, \
         patch("backend.services.place_context.httpx.AsyncClient", return_value=_mock_client(payload)):
        res = asyncio.run(fetch_place_context("97028"))

    assert res["status"] == "present"
    assert res["population"] == 230
    assert res["rural"] is True
    set_cache.assert_called_once_with("pop:97028", res)


def test_fetch_place_context_urban_large_pop():
    payload = [["NAME", "P1_001N", "zip code tabulation area"], ["ZCTA5 90012", "223000", "90012"]]
    with patch.object(place_context, "CENSUS_API_KEY", "test-key"), \
         patch.object(place_context, "get_cached_place", return_value=None), \
         patch.object(place_context, "set_cached_place"), \
         patch("backend.services.place_context.httpx.AsyncClient", return_value=_mock_client(payload)):
        res = asyncio.run(fetch_place_context("90012"))

    assert res["status"] == "present"
    assert res["population"] == 223000
    assert res["rural"] is False


def test_fetch_place_context_uses_cache_without_network():
    cached = {"status": "present", "population": 230, "rural": True, "details": "ZCTA population 230"}
    with patch.object(place_context, "get_cached_place", return_value=cached) as get_cache, \
         patch("backend.services.place_context.httpx.AsyncClient") as client_cls:
        res = asyncio.run(fetch_place_context("97028"))

    assert res == cached
    get_cache.assert_called_once_with("pop:97028", max_age_days=365)
    client_cls.assert_not_called()


def test_fetch_place_context_no_key_is_unavailable():
    with patch.object(place_context, "CENSUS_API_KEY", ""), \
         patch.object(place_context, "get_cached_place", return_value=None), \
         patch("backend.services.place_context.httpx.AsyncClient") as client_cls:
        res = asyncio.run(fetch_place_context("97028"))

    assert res["status"] == "unavailable"
    assert res["population"] is None
    assert res["rural"] is None
    client_cls.assert_not_called()


def test_fetch_place_context_failure_is_unavailable():
    with patch.object(place_context, "CENSUS_API_KEY", "test-key"), \
         patch.object(place_context, "get_cached_place", return_value=None), \
         patch.object(place_context, "set_cached_place"), \
         patch("backend.services.place_context.httpx.AsyncClient", return_value=_mock_client(raise_exc=True)):
        res = asyncio.run(fetch_place_context("97028"))

    assert res["status"] == "unavailable"
    assert res["rural"] is None


def test_fetch_place_context_strips_zip_plus_four():
    """A ZIP+4 (97028-1234) must be truncated to the 5-digit ZCTA before querying."""
    payload = [["NAME", "P1_001N", "zip code tabulation area"], ["ZCTA5 97028", "230", "97028"]]
    mock_client = _mock_client(payload)
    with patch.object(place_context, "CENSUS_API_KEY", "test-key"), \
         patch.object(place_context, "get_cached_place", return_value=None), \
         patch.object(place_context, "set_cached_place") as set_cache, \
         patch("backend.services.place_context.httpx.AsyncClient", return_value=mock_client):
        res = asyncio.run(fetch_place_context("97028-1234"))

    assert res["status"] == "present"
    params = mock_client.get.call_args.kwargs["params"]
    assert params["for"] == "zip code tabulation area:97028"
    set_cache.assert_called_once_with("pop:97028", res)


def test_fetch_place_context_uses_dhc_endpoint():
    """ZCTA population must come from dec/dhc (dec/pl rejects the geography)."""
    payload = [["NAME", "P1_001N", "zip code tabulation area"], ["ZCTA5 98801", "44970", "98801"]]
    mock_client = _mock_client(payload)
    with patch.object(place_context, "CENSUS_API_KEY", "test-key"), \
         patch.object(place_context, "get_cached_place", return_value=None), \
         patch.object(place_context, "set_cached_place"), \
         patch("backend.services.place_context.httpx.AsyncClient", return_value=mock_client):
        asyncio.run(fetch_place_context("98801"))

    url = mock_client.get.call_args.args[0]
    assert "/2020/dec/dhc" in url


def test_fetch_place_context_no_zip_is_unavailable():
    with patch("backend.services.place_context.httpx.AsyncClient") as client_cls:
        res = asyncio.run(fetch_place_context(None))

    assert res["status"] == "unavailable"
    client_cls.assert_not_called()
