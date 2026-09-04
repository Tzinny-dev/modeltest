"""Shared helpers used by built-in scenarios."""
from __future__ import annotations

from typing import Any

import numpy as np

_EPS = 1e-6


def model_features(model: Any, X: Any) -> Any:
    """Return the subset of columns the model was trained on, if discernible.

    Validation data often carries extra columns (grouping keys, ids) that the
    model never saw at fit time. ``sklearn`` exposes ``feature_names_in_`` so we
    slice to those columns when possible, letting prediction succeed.
    """
    fn = getattr(model, "feature_names_in_", None)
    if fn is not None and hasattr(X, "columns"):
        return X[[c for c in fn if c in X.columns]]
    return X


def psi(expected: Any, actual: Any, n_bins: int = 10) -> float:
    """Population Stability Index between two 1-D samples.

    Bins are computed from the union of both samples' quantiles so that a
    ``reference`` vs ``current`` comparison is stable and comparable. A small
    floor avoids log(0) / div-by-zero.

    Interpretation:
        PSI < 0.1  -> negligible shift
        PSI < 0.25 -> moderate shift (flag)
        PSI >= 0.25-> significant shift (action)
    """
    expected = np.asarray(expected, dtype=float).ravel()
    actual = np.asarray(actual, dtype=float).ravel()
    if expected.size == 0 or actual.size == 0:
        return 0.0

    lo = min(float(expected.min()), float(actual.min()))
    hi = max(float(expected.max()), float(actual.max()))
    if lo == hi:
        return 0.0

    edges = np.linspace(lo, hi, n_bins + 1)
    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)

    exp_pct = exp_counts / max(expected.size, 1) + _EPS
    act_pct = act_counts / max(actual.size, 1) + _EPS

    ratio = act_pct / exp_pct
    return float(np.sum((act_pct - exp_pct) * np.log(ratio)))


def ks_statistic(reference: Any, current: Any) -> float:
    """Two-sample Kolmogorov–Smirnov D statistic (largest ECDF gap)."""
    from scipy import stats

    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()
    if ref.size == 0 or cur.size == 0:
        return 0.0
    d, _ = stats.ks_2samp(ref, cur)
    return float(d)


def group_rates(
    y_true: Any, y_pred: Any, protected: Any, pos_label: int = 1
) -> "dict[str, dict[str, float]]":
    """Per-group fairness metrics: selection rate and true positive rate.

    Returns ``{group_label: {"selection_rate": s, "tpr": t}}``. ``protected``
    is an array-like of group labels (one per sample).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    protected = np.asarray(protected)

    rates: dict[str, dict[str, float]] = {}
    for g in np.unique(protected):
        mask = protected == g
        group_true = y_true[mask]
        group_pred = y_pred[mask]

        selection = float(np.mean(group_pred == pos_label))
        positives = group_true == pos_label
        tpr = (
            float(np.mean(group_pred[positives] == pos_label))
            if positives.sum() > 0
            else 1.0
        )
        rates[str(g)] = {"selection_rate": selection, "tpr": tpr}
    return rates


def max_pairwise_gap(values: "dict[str, float]") -> float:
    """Largest difference between any two values in a per-group mapping."""
    vals = list(values.values())
    return float(max(vals) - min(vals)) if vals else 0.0


def min_max_ratio(values: "dict[str, float]") -> float:
    """Smallest-to-largest ratio of a per-group mapping (for disparate impact)."""
    vals = [v for v in values.values() if not np.isclose(v, 0)]
    if len(vals) < 2:
        return 1.0
    return float(min(vals) / max(vals))
