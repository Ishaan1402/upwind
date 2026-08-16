"""Tests for the pure classification metrics layer (Phase 1d-1).

Hand-computed expected values on a small known set, including ambiguous rows
that must be excluded from per-class metrics but kept in the confusion matrix.
"""

import pytest

from backend.eval.accuracy.labels import LABEL_CLASSES
from backend.eval.accuracy.metrics import METRIC_NOTES, compute_metrics

# ---------------------------------------------------------------------------
# Hand-computed fixture
# ---------------------------------------------------------------------------
# 7 samples: 5 coverage (true != "ambiguous") + 2 ambiguous.
#   wildfire_smoke -> wildfire_smoke   TP smoke, and the only smoke TP
#   wildfire_smoke -> ozone_episode    FN smoke / FP ozone
#   clean          -> clean            TP clean
#   clean          -> wildfire_smoke   FN clean / FP smoke
#   windblown_dust -> windblown_dust   TP dust
#   ambiguous      -> clean            excluded from per-class metrics
#   ambiguous      -> ambiguous        excluded from per-class metrics
FIXTURE_RESULTS = [
    ("wildfire_smoke", "wildfire_smoke"),
    ("wildfire_smoke", "ozone_episode"),
    ("clean", "clean"),
    ("clean", "wildfire_smoke"),
    ("windblown_dust", "windblown_dust"),
    ("ambiguous", "clean"),
    ("ambiguous", "ambiguous"),
]


def test_metrics_top_level_counts():
    metrics = compute_metrics(FIXTURE_RESULTS)
    assert metrics["total"] == 7
    assert metrics["ambiguous_count"] == 2
    assert metrics["coverage"] == 5
    assert metrics["top1_accuracy"] == pytest.approx(3 / 5)  # 3 correct of 5


def test_metrics_clean_and_elevated_counts():
    metrics = compute_metrics(FIXTURE_RESULTS)
    # True labels: smoke x2, clean x2, dust x1, ambiguous x2.
    assert metrics["clean_count"] == 2
    assert metrics["elevated_count"] == 3
    # Elevated samples: (smoke, smoke) correct, (smoke, ozone) wrong,
    # (dust, dust) correct -> 2/3. Clean days are excluded so the AQI<=50
    # mapping cannot inflate this number.
    assert metrics["non_clean_top1_accuracy"] == pytest.approx(2 / 3)


def test_metrics_notes_are_present():
    metrics = compute_metrics(FIXTURE_RESULTS)
    assert metrics["notes"] == list(METRIC_NOTES)
    assert METRIC_NOTES[0] == (
        "labels are rule-derived (self-consistency, not independent ground truth)"
    )
    assert METRIC_NOTES[1] == "clean predictions use an AQI<=50 mapping, not a learned model"


def test_metrics_confusion_spans_all_labels():
    metrics = compute_metrics(FIXTURE_RESULTS)
    confusion = metrics["confusion"]
    # Both axes cover all 7 label classes (zero-filled where absent).
    assert set(confusion.keys()) == set(LABEL_CLASSES)
    assert all(set(row.keys()) == set(LABEL_CLASSES) for row in confusion.values())

    assert confusion["wildfire_smoke"]["wildfire_smoke"] == 1
    assert confusion["wildfire_smoke"]["ozone_episode"] == 1
    assert confusion["clean"]["clean"] == 1
    assert confusion["clean"]["wildfire_smoke"] == 1
    assert confusion["windblown_dust"]["windblown_dust"] == 1
    # Ambiguous rows are present in the matrix...
    assert confusion["ambiguous"]["clean"] == 1
    assert confusion["ambiguous"]["ambiguous"] == 1
    # ...and everything else is zero.
    assert confusion["wildfire_smoke"]["windblown_dust"] == 0
    assert confusion["winter_stagnation"]["winter_stagnation"] == 0


def test_metrics_per_class_precision_recall_f1():
    metrics = compute_metrics(FIXTURE_RESULTS)
    per_class = metrics["per_class"]

    # TP=1, FP=1 (clean->smoke), FN=1 (smoke->ozone) -> 0.5 / 0.5 / 0.5.
    smoke = per_class["wildfire_smoke"]
    assert smoke["precision"] == pytest.approx(0.5)
    assert smoke["recall"] == pytest.approx(0.5)
    assert smoke["f1"] == pytest.approx(0.5)

    # TP=1, FP=0, FN=1 (clean->smoke) -> 1.0 / 0.5 / 2/3.
    clean = per_class["clean"]
    assert clean["precision"] == pytest.approx(1.0)
    assert clean["recall"] == pytest.approx(0.5)
    assert clean["f1"] == pytest.approx(2 / 3)

    # Perfect row.
    dust = per_class["windblown_dust"]
    assert dust == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    # Never predicted as ozone -> precision 0.0; never a true ozone sample ->
    # recall None (zero denominator); F1 therefore None.
    ozone = per_class["ozone_episode"]
    assert ozone["precision"] == 0.0
    assert ozone["recall"] is None
    assert ozone["f1"] is None

    # No samples at all -> every metric None (zero denominators).
    for cls in ("winter_stagnation", "urban_industrial_pm"):
        assert per_class[cls] == {"precision": None, "recall": None, "f1": None}


def test_metrics_macro_f1_mean_of_non_none():
    metrics = compute_metrics(FIXTURE_RESULTS)
    expected = (0.5 + (2 / 3) + 1.0) / 3
    assert metrics["macro_f1"] == pytest.approx(expected)


def test_metrics_macro_f1_none_without_any_f1():
    # A single misclassified sample gives no class a computable F1 (each side
    # has a zero denominator somewhere).
    metrics = compute_metrics([("winter_stagnation", "ozone_episode")])
    assert metrics["macro_f1"] is None
    assert metrics["top1_accuracy"] == 0.0
    assert metrics["non_clean_top1_accuracy"] == 0.0
    assert metrics["coverage"] == 1
    assert metrics["ambiguous_count"] == 0
    assert metrics["clean_count"] == 0
    assert metrics["elevated_count"] == 1


def test_metrics_empty_results():
    metrics = compute_metrics([])
    assert metrics["total"] == 0
    assert metrics["ambiguous_count"] == 0
    assert metrics["clean_count"] == 0
    assert metrics["elevated_count"] == 0
    assert metrics["coverage"] == 0
    assert metrics["macro_f1"] is None
    assert metrics["top1_accuracy"] is None
    assert metrics["non_clean_top1_accuracy"] is None
    assert metrics["notes"] == list(METRIC_NOTES)
    assert all(
        per["precision"] is None and per["recall"] is None and per["f1"] is None
        for per in metrics["per_class"].values()
    )


def test_metrics_all_ambiguous_rows_no_coverage():
    metrics = compute_metrics([("ambiguous", "clean"), ("ambiguous", "urban_industrial_pm")])
    assert metrics["total"] == 2
    assert metrics["ambiguous_count"] == 2
    assert metrics["clean_count"] == 0
    assert metrics["elevated_count"] == 0
    assert metrics["coverage"] == 0
    assert metrics["top1_accuracy"] is None
    assert metrics["non_clean_top1_accuracy"] is None
    assert metrics["macro_f1"] is None
    assert metrics["confusion"]["ambiguous"]["clean"] == 1
