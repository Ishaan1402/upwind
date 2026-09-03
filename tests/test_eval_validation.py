from backend.eval_validation import (
    VALIDATION_SCENARIOS,
    _cohens_kappa,
    build_validation_set,
)


def test_validation_set_has_two_cases_per_scenario():
    cases = build_validation_set()
    assert len(VALIDATION_SCENARIOS) >= 15
    assert len(cases) == len(VALIDATION_SCENARIOS) * 2 >= 30
    assert all(c["gold_verdict"] in ("pass", "fail") for c in cases)


def test_every_scenario_has_grounded_and_hallucinated_case():
    cases = build_validation_set()
    for scenario in VALIDATION_SCENARIOS:
        names = [c["name"] for c in cases if c["name"].startswith(scenario["name"])]
        assert any(name.endswith("_grounded") for name in names)
        assert any(name.endswith("_hallucinated") for name in names)


def test_cohens_kappa_perfect_and_random():
    perfect = ["pass", "pass", "fail", "fail"]
    assert _cohens_kappa(perfect, perfect) == 1.0
    assert _cohens_kappa([], []) is None
