"""Unit tests for the Phase-2 coordinate-search tuning module.

These cover the pure building blocks of ``backend.eval.accuracy.tune``: the
deterministic site-based holdout split, the ``replace(Params(), **overrides)``
grid-combo construction, and the prediction mapping shared with the runner.
The end-to-end ``run_tune`` sweep is exercised by the smoke-run against the
real store rather than a synthetic fixture here.
"""

import dataclasses

from backend.engine.params import Params
from backend.eval.accuracy.tune import (
    PHASE2_GRID,
    _predicted_label,
    _site_holdout_split,
)


def _sector45_overrides():
    """The ``sector45`` grid overrides, looked up by name (not by index)."""
    return next(overrides for name, overrides in PHASE2_GRID if name == "sector45")


def test_site_holdout_split_is_deterministic_and_respects_fraction():
    site_ids = [f"06-037-{i:04d}" for i in range(100)]

    train_1, val_1 = _site_holdout_split(site_ids, val_fraction=0.2)
    train_2, val_2 = _site_holdout_split(site_ids, val_fraction=0.2)

    # Re-running over the same site list gives an identical split.
    assert train_1 == train_2
    assert val_1 == val_2

    # ~20% of 100 sites land in val (md5 buckets are near-uniform).
    assert 15 <= len(val_1) <= 25
    assert len(train_1) + len(val_1) == len(site_ids)

    # A site is entirely in one partition: no site-day leakage across the
    # train/val boundary.
    assert set(train_1).isdisjoint(val_1)

    # The split is site-based, so site-days of one site stay together.
    same_site_days = ["06-037-0001", "06-037-0001", "06-037-0002", "06-037-0002"]
    train_days, val_days = _site_holdout_split(same_site_days, val_fraction=0.5)
    for site in {"06-037-0001", "06-037-0002"}:
        assert (site in train_days) != (site in val_days)


def test_grid_override_yields_params_matching_defaults_elsewhere():
    overrides = _sector45_overrides()
    params = dataclasses.replace(Params(), **overrides)

    assert isinstance(params, Params)
    assert params.upwind_sector_width_deg == 45.0
    # Only the overridden field differs: rebuilding the frozen Params() with
    # the same explicit override must reproduce the combo's params exactly.
    expected = dataclasses.replace(Params(), upwind_sector_width_deg=45.0)
    assert params == expected


def test_predicted_label_mapping():
    # AQI at/below the elevated threshold is always clean, whatever the scorer
    # ranks first (the scorer has no "clean" hypothesis id).
    assert _predicted_label({"aqi": 50}, [{"id": "wildfire_smoke"}]) == "clean"
    assert _predicted_label({"aqi": 30}, [{"id": "windblown_dust"}]) == "clean"

    # A missing AQI is NOT clean: empty hypotheses -> ambiguous...
    assert _predicted_label({"aqi": None}, []) == "ambiguous"
    # ...and non-empty hypotheses -> the top hypothesis id.
    assert _predicted_label({"aqi": None}, [{"id": "windblown_dust"}]) == "windblown_dust"

    # Elevated AQI falls through to the top hypothesis.
    assert _predicted_label({"aqi": 120}, [{"id": "windblown_dust"}]) == "windblown_dust"
    assert _predicted_label({"aqi": 120}, [{"id": "wildfire_smoke"}]) == "wildfire_smoke"
