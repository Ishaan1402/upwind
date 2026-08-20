"""Tests for the NWS dust-alert confirmation feed (backend/services/nws.py)."""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch

from backend.services.nws import (
    fetch_dust_alert,
    _find_dust_alert,
    _headline,
    DUST_EVENT_NAMES,
)


def _alert_feature(event, headline=None, severity="Moderate", extra_params=None):
    props = {"event": event, "severity": severity}
    if headline is not None:
        props["headline"] = headline
    if extra_params:
        props["parameters"] = extra_params
    return {"type": "Feature", "properties": props}


def _feature_collection(features):
    return {"type": "FeatureCollection", "features": features}


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


def test_dust_event_names_are_exact_expected_strings():
    """Guard the exact event-name substrings wired into the confirmation."""
    assert DUST_EVENT_NAMES == (
        "Dust Storm Warning",
        "Blowing Dust Warning",
        "Dust Advisory",
        "Blowing Dust Advisory",
    )


def test_find_dust_alert_matches_each_event_name_case_insensitively():
    for name in DUST_EVENT_NAMES:
        # Event names can arrive in mixed case; matching is case-insensitive.
        props = _find_dust_alert([_alert_feature(name.lower())])
        assert props is not None
        assert props["event"] == name.lower()


def test_find_dust_alert_substring_matches_within_longer_event():
    """An event like 'Blowing Dust Advisory in Effect until 7 PM MDT' still matches."""
    props = _find_dust_alert(
        [_alert_feature("Blowing Dust Advisory in Effect until 7 PM MDT")]
    )
    assert props is not None
    assert props["event"].startswith("Blowing Dust Advisory")


def test_find_dust_alert_returns_none_without_dust_event():
    assert _find_dust_alert([_alert_feature("Extreme Heat Warning")]) is None
    assert _find_dust_alert([]) is None
    assert _find_dust_alert([{"type": "Feature", "properties": {}}]) is None


def test_headline_falls_back_to_parameters():
    feature = _alert_feature("Dust Storm Warning")
    assert _headline(feature["properties"]) is None
    feature = _alert_feature(
        "Dust Storm Warning", extra_params={"NWSheadline": ["DUST STORM WARNING IN EFFECT"]}
    )
    assert _headline(feature["properties"]) == "DUST STORM WARNING IN EFFECT"


def test_fetch_dust_alert_present_returns_event_and_headline():
    payload = _feature_collection([
        _alert_feature("Extreme Heat Warning", headline="Heat warning", severity="Severe"),
        _alert_feature(
            "Dust Storm Warning",
            headline="Dust Storm Warning issued for the El Paso area",
            severity="Severe",
        ),
    ])
    with patch("backend.services.nws.httpx.AsyncClient", return_value=_mock_client(payload)):
        res = asyncio.run(fetch_dust_alert(31.76, -106.49))

    assert res["status"] == "present"
    assert res["event"] == "Dust Storm Warning"
    assert res["headline"] == "Dust Storm Warning issued for the El Paso area"
    assert res["severity"] == "Severe"


def test_fetch_dust_alert_absent_when_no_dust_event():
    payload = _feature_collection([
        _alert_feature("Extreme Heat Warning", headline="Heat warning"),
        _alert_feature("Air Quality Alert", headline="Ozone advisory"),
    ])
    with patch("backend.services.nws.httpx.AsyncClient", return_value=_mock_client(payload)):
        res = asyncio.run(fetch_dust_alert(33.45, -112.07))

    assert res["status"] == "absent"
    assert res["event"] is None
    assert res["headline"] is None


def test_fetch_dust_alert_absent_on_empty_feed():
    with patch("backend.services.nws.httpx.AsyncClient", return_value=_mock_client(_feature_collection([]))):
        res = asyncio.run(fetch_dust_alert(31.76, -106.49))
    assert res["status"] == "absent"
    assert res["event"] is None


def test_fetch_dust_alert_unavailable_on_http_error():
    with patch("backend.services.nws.httpx.AsyncClient", return_value=_mock_client(status=500)):
        res = asyncio.run(fetch_dust_alert(31.76, -106.49))
    assert res["status"] == "unavailable"
    assert res["event"] is None
    assert "HTTP 500" in res["details"]


def test_fetch_dust_alert_unavailable_on_exception():
    with patch("backend.services.nws.httpx.AsyncClient", return_value=_mock_client(raise_exc=True)):
        res = asyncio.run(fetch_dust_alert(31.76, -106.49))
    assert res["status"] == "unavailable"
    assert res["event"] is None
    assert res["headline"] is None
    assert "details" in res


def test_fetch_dust_alert_unavailable_on_unexpected_payload():
    with patch("backend.services.nws.httpx.AsyncClient", return_value=_mock_client(["not", "a", "dict"])):
        res = asyncio.run(fetch_dust_alert(31.76, -106.49))
    assert res["status"] == "unavailable"
