"""Performance tests: global and per-subgroup quality thresholds."""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from modeltest.core.base import ModelTest, TestContext
from modeltest.scenarios._utils import model_features

_METRICS = {
    "accuracy": lambda yt, yp: float(np.mean(np.asarray(yt) == np.asarray(yp))),
    "precision": None,  # filled lazily to avoid importing sklearn unconditionally
    "recall": None,
    "f1": None,
}


def _sk_metric(name: str) -> Callable:
    from sklearn import metrics as _m

    return {
        "precision": _m.precision_score,
        "recall": _m.recall_score,
        "f1": _m.f1_score,
        "roc_auc": _m.roc_auc_score,
    }[name]


def resolve_metric(metric: str) -> Callable[[Any, Any], float]:
    name = metric.lower()
    if name in _METRICS:
        fn = _METRICS[name]
        return _sk_metric(name) if fn is None else fn
    if name in {"precision", "recall", "f1", "roc_auc"}:
        return _sk_metric(name)
    raise ValueError(f"Unknown metric: {metric}")


class MinimumAccuracyTest(ModelTest):
    """Assert overall accuracy is at least a threshold."""

    def __init__(self, threshold: float = 0.85, metric: str = "accuracy"):
        self.threshold = threshold
        self.metric = metric

    def test(self, ctx: TestContext) -> Any:
        fn = resolve_metric(self.metric)
        score = fn(ctx.y_val, ctx.model.predict(model_features(ctx.model, ctx.X_val)))
        assert score >= self.threshold, (
            f"{self.metric} = {score:.4f} < threshold {self.threshold}"
        )


class GroupPerformanceTest(ModelTest):
    """Assert a metric stays above a threshold for every subgroup.

    ``group_col`` names a categorical column in ``X_val``; the metric is
    evaluated per group and all must meet the threshold.
    """

    def __init__(
        self,
        metric: str = "accuracy",
        threshold: float = 0.8,
        group_col: str = "gender",
    ):
        self.metric = metric
        self.threshold = threshold
        self.group_col = group_col

    def test(self, ctx: TestContext) -> Any:
        fn = resolve_metric(self.metric)
        y_pred = np.asarray(ctx.model.predict(model_features(ctx.model, ctx.X_val)))
        y_true = np.asarray(ctx.y_val)
        groups = ctx.X_val[self.group_col].astype(str)
        for g in np.unique(groups):
            mask = groups == g
            score = fn(y_true[mask], y_pred[mask])
            assert score >= self.threshold, (
                f"group {g!r}: {self.metric} = {score:.4f} < threshold {self.threshold}"
            )
