import time

from unittest.mock import patch

from backend.observation_token import sign_observation, verify_observation_token


LOCATION = {"lat": 34.09, "lon": -118.41, "name": "Beverly Hills"}
OBSERVATION = {"aqi": 42, "primary_pollutant": "PM2.5", "category": "Good"}


def test_round_trip():
    token = sign_observation(LOCATION, OBSERVATION, "secret")
    assert token is not None
    assert verify_observation_token(token, LOCATION, OBSERVATION, "secret") is True


def test_tampered_token_rejected():
    token = sign_observation(LOCATION, OBSERVATION, "secret")
    tampered = token[:-2] + ("ab" if not token.endswith("ab") else "cd")
    assert verify_observation_token(tampered, LOCATION, OBSERVATION, "secret") is False


def test_wrong_secret_rejected():
    token = sign_observation(LOCATION, OBSERVATION, "secret")
    assert verify_observation_token(token, LOCATION, OBSERVATION, "other") is False


def test_mismatched_observation_rejected():
    token = sign_observation(LOCATION, OBSERVATION, "secret")
    changed = {**OBSERVATION, "aqi": 99}
    assert verify_observation_token(token, LOCATION, changed, "secret") is False


def test_expired_token_rejected(monkeypatch):
    import backend.observation_token as token_module

    token = sign_observation(LOCATION, OBSERVATION, "secret", max_age_seconds=1)
    future = time.time() + 60
    monkeypatch.setattr(token_module.time, "time", lambda: future)
    assert verify_observation_token(token, LOCATION, OBSERVATION, "secret") is False


def test_why_post_requires_token_when_enforced():
    from backend.main import app
    from fastapi.testclient import TestClient

    with patch("backend.routers.why.ENFORCE_OBSERVATION_TOKENS", True), \
         patch("backend.routers.why.OBSERVATION_TOKEN_SECRET", "secret"):
        with TestClient(app) as client:
            resp = client.post(
                "/api/why",
                json={"location": LOCATION, "observation": OBSERVATION},
            )
    assert resp.status_code == 400
    assert "observation_token" in resp.json()["detail"]
