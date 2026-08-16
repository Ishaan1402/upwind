"""Pure classification metrics for the accuracy-evaluation pipeline.

This module is I/O free: it turns a list of ``(true_label, predicted_label)``
pairs into a confusion matrix plus standard per-class and aggregate metrics.
Labels are the ``labels.LABEL_CLASSES`` classes (five scorer hypothesis ids,
``"clean"``, and ``"ambiguous"``).

Samples whose ground truth is ``"ambiguous"`` are kept in the confusion matrix
(``total``/``ambiguous_count`` reflect every sample) but excluded from the
per-class precision/recall/F1 and from ``top1_accuracy``, since an ambiguous
ground truth has no single correct attribution. Per-class metrics use the
standard definitions over that coverage subset; any metric whose denominator
is zero is ``None`` (not a crash and not a fabricated 0).

``notes`` states the honesty caveats baked into the metric set: the labels are
rule-derived (self-consistency, not independent ground truth) and clean
predictions come from the AQI<=50 mapping in the runner, not a learned model.
"""

from typing import Dict, Iterable, List, Optional, Tuple

from backend.eval.accuracy.labels import LABEL_CLASSES

# Labels with a defensible per-class metric: the five scorer hypothesis ids
# plus "clean". "ambiguous" is excluded from per-class metrics.
_NON_AMBIGUOUS_CLASSES: Tuple[str, ...] = tuple(c for c in LABEL_CLASSES if c != "ambiguous")

# Honesty caveats that must accompany every metric summary (see module docstring).
METRIC_NOTES: Tuple[str, ...] = (
    "labels are rule-derived (self-consistency, not independent ground truth)",
    "clean predictions use an AQI<=50 mapping, not a learned model",
)


def _safe_ratio(numerator: int, denominator: int) -> Optional[float]:
    """``numerator / denominator`` as a float, or None on a zero denominator."""
    if denominator == 0:
        return None
    return numerator / denominator


def compute_metrics(results: Iterable[Tuple[str, str]]) -> Dict:
    """Compute accuracy metrics over ``(true_label, predicted_label)`` pairs.

    Returns:
        confusion:      nested ``true_label -> {predicted_label -> count}``
                        spanning all 7 ``LABEL_CLASSES`` on both axes.
        total:          number of samples evaluated.
        ambiguous_count: samples whose true label is ``"ambiguous"``.
        clean_count:    samples whose true label is ``"clean"``.
        elevated_count: samples whose true label is neither ``"clean"`` nor
                        ``"ambiguous"`` (the attributable elevated classes).
        coverage:       samples whose true label is not ``"ambiguous"``.
        per_class:      for each non-ambiguous label,
                        ``{"precision", "recall", "f1"}`` computed over the
                        coverage subset (None on a zero denominator).
        macro_f1:       unweighted mean of the non-None per-class F1 values
                        (None when no class has a computable F1).
        top1_accuracy:  fraction of coverage samples whose predicted label
                        equals the true label (None when coverage is 0).
        non_clean_top1_accuracy: fraction of elevated samples whose predicted
                        label equals the true label (None when there are no
                        elevated samples) — clean days excluded so the AQI<=50
                        mapping cannot inflate this number.
        notes:          honesty caveats (``METRIC_NOTES``) explaining that the
                        labels are rule-derived and clean predictions are a
                        threshold mapping.
    """
    rows: List[Tuple[str, str]] = list(results)

    confusion: Dict[str, Dict[str, int]] = {
        true: {pred: 0 for pred in LABEL_CLASSES} for true in LABEL_CLASSES
    }
    for true_label, predicted_label in rows:
        # Tolerate labels outside LABEL_CLASSES so callers can feed raw scorer
        # output without pre-validation.
        confusion.setdefault(true_label, {p: 0 for p in LABEL_CLASSES})
        confusion[true_label].setdefault(predicted_label, 0)
        confusion[true_label][predicted_label] += 1

    total = len(rows)
    ambiguous_count = sum(1 for true_label, _ in rows if true_label == "ambiguous")
    clean_count = sum(1 for true_label, _ in rows if true_label == "clean")
    coverage_rows = [row for row in rows if row[0] != "ambiguous"]
    coverage = len(coverage_rows)
    elevated_rows = [row for row in rows if row[0] != "clean" and row[0] != "ambiguous"]
    elevated_count = len(elevated_rows)

    per_class: Dict[str, Dict[str, Optional[float]]] = {}
    for cls in _NON_AMBIGUOUS_CLASSES:
        tp = sum(1 for true_label, predicted_label in coverage_rows if true_label == cls and predicted_label == cls)
        fp = sum(1 for true_label, predicted_label in coverage_rows if true_label != cls and predicted_label == cls)
        fn = sum(1 for true_label, predicted_label in coverage_rows if true_label == cls and predicted_label != cls)

        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = None
        if precision is not None and recall is not None and precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1}

    f1_values = [per_class[cls]["f1"] for cls in _NON_AMBIGUOUS_CLASSES if per_class[cls]["f1"] is not None]
    macro_f1 = (sum(f1_values) / len(f1_values)) if f1_values else None

    top1 = sum(1 for true_label, predicted_label in coverage_rows if true_label == predicted_label)
    top1_accuracy = _safe_ratio(top1, coverage)

    non_clean_top1 = sum(
        1 for true_label, predicted_label in elevated_rows if true_label == predicted_label
    )
    non_clean_top1_accuracy = _safe_ratio(non_clean_top1, elevated_count)

    return {
        "confusion": confusion,
        "total": total,
        "ambiguous_count": ambiguous_count,
        "clean_count": clean_count,
        "elevated_count": elevated_count,
        "coverage": coverage,
        "per_class": per_class,
        "macro_f1": macro_f1,
        "top1_accuracy": top1_accuracy,
        "non_clean_top1_accuracy": non_clean_top1_accuracy,
        "notes": list(METRIC_NOTES),
    }
