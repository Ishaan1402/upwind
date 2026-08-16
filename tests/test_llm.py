"""Deterministic fallback narrative regression tests."""

from backend.llm import generate_fallback_narrative

LOC = {"name": "Test Town"}
GOOD_OBS = {"aqi": 35, "primary_pollutant": "PM2.5", "category": "Good"}


def _signals(*present_ids):
    signals = []
    for sid in ("aerosol_plume", "firms_upwind", "hms_smoke", "wfigs_incident"):
        signals.append({"id": sid, "status": "present" if sid in present_ids else "absent"})
    return signals


def test_good_aqi_fallback_leads_with_clean_air():
    text = generate_fallback_narrative(LOC, GOOD_OBS, _signals(), [], [])
    assert text.startswith("Air quality in Test Town is currently Good")
    assert "aloft" not in text


def test_good_aqi_hms_plume_gets_aloft_context():
    text = generate_fallback_narrative(LOC, GOOD_OBS, _signals("hms_smoke"), [], [])
    assert "clean and healthy" in text
    assert "aloft" in text


def test_good_aqi_wfigs_incident_gets_aloft_context():
    text = generate_fallback_narrative(LOC, GOOD_OBS, _signals("wfigs_incident"), [], [])
    assert "clean and healthy" in text
    assert "aloft" in text
