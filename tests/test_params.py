"""Unit tests for the frozen Params dataclass and its contextvar accessor."""

import dataclasses

import pytest

from backend.engine.params import DEFAULT, LABEL_PARAMS, Params, get_params, use_params


def test_params_frozen_distinct_label_and_contextvar():
    # DEFAULT and LABEL_PARAMS are distinct frozen instances; DEFAULT carries
    # the Phase-2 tuned thresholds while LABEL_PARAMS stays frozen at the
    # original values.
    assert DEFAULT is not LABEL_PARAMS
    assert DEFAULT != LABEL_PARAMS
    assert isinstance(DEFAULT, Params)
    assert isinstance(LABEL_PARAMS, Params)
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT.aqi_elevated = 99
    with pytest.raises(dataclasses.FrozenInstanceError):
        LABEL_PARAMS.aqi_elevated = 99

    # Phase-2 tuning locked the dust-loft wind divergence: DEFAULT is tuned
    # down to 12 mph while LABEL_PARAMS keeps the original 15.0 (frozen).
    assert LABEL_PARAMS.high_wind_speed_mph == 15.0
    assert DEFAULT.high_wind_speed_mph == 12.0
    assert DEFAULT.high_wind_speed_mph != LABEL_PARAMS.high_wind_speed_mph

    # get_params() returns DEFAULT by default, use_params() swaps it for the
    # block and reverts afterwards (including when an exception is raised).
    assert get_params() is DEFAULT
    custom = Params(upwind_sector_width_deg=45.0)
    with use_params(custom):
        assert get_params() is custom
    assert get_params() is DEFAULT
    with pytest.raises(RuntimeError):
        with use_params(custom):
            assert get_params() is custom
            raise RuntimeError("boom")
    assert get_params() is DEFAULT

    # Spot-check that the preserved values match the pre-refactor constants.
    assert DEFAULT.upwind_sector_width_deg == 90.0
    assert DEFAULT.wfigs_upwind_bonus == 4.0
    assert DEFAULT.aqi_elevated == 50
    assert dict(DEFAULT.firms_confidence_weight) == {"low": 0.0, "nominal": 0.7, "high": 1.0}
    with pytest.raises(TypeError):
        DEFAULT.firms_confidence_weight["low"] = 1.0
    assert LABEL_PARAMS.upwind_sector_width_deg == 90.0
    assert LABEL_PARAMS.wfigs_upwind_bonus == 4.0
    assert LABEL_PARAMS.aqi_elevated == 50
    assert dict(LABEL_PARAMS.firms_confidence_weight) == {"low": 0.0, "nominal": 0.7, "high": 1.0}
    with pytest.raises(TypeError):
        LABEL_PARAMS.firms_confidence_weight["low"] = 1.0
