"""Tests for bootstrap confidence-interval thresholds."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from modeltest import TestContext
from modeltest.core.base import TestStatus
from modeltest.scenarios import ConfidenceThresholdTest
from modeltest.scenarios._utils import bootstrap_ci
from modeltest.scenarios.performance import resolve_metric


class TestBootstrapCI:
    def test_returns_estimate_and_interval(self):
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        estimate, lower, upper = bootstrap_ci(
            y_true, y_pred, resolve_metric("accuracy"), n_boot=200
        )
        assert 0.0 <= estimate <= 1.0
        assert 0.0 <= lower <= estimate <= upper <= 1.0
        assert lower < upper

    def test_interval_shrinks_with_more_data(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 2000)
        y_pred = rng.integers(0, 2, 2000)
        fn = resolve_metric("accuracy")
        _, l_small, u_small = bootstrap_ci(
            y_true[:50], y_pred[:50], fn, n_boot=150, random_state=1
        )
        _, l_large, u_large = bootstrap_ci(
            y_true, y_pred, fn, n_boot=150, random_state=1
        )
        assert (u_large - l_large) < (u_small - l_small)

    def test_rejects_empty_sample(self):
        import pytest

        with pytest.raises(ValueError):
            bootstrap_ci([], [], resolve_metric("accuracy"))


def _make_ctx(n=600):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "age": rng.normal(45, 12, n),
            "income": rng.normal(60000, 20000, n),
        }
    )
    y = (X["income"] > X["age"] * 800 + rng.normal(0, 16000, n)).astype(int)
    model = RandomForestClassifier(n_estimators=40, random_state=0).fit(X, y)
    return TestContext(model=model, X_val=X, y_val=y)


class TestConfidenceThreshold:
    def test_passes_when_lower_bound_clears_threshold(self):
        ctx = _make_ctx()
        result = ConfidenceThresholdTest(
            metric="accuracy", threshold=0.0, n_boot=100
        ).run(ctx)
        assert result.status == TestStatus.PASSED

    def test_fails_when_lower_bound_below_threshold(self):
        rng = np.random.default_rng(1)
        n = 600
        X = pd.DataFrame(
            {
                "age": rng.normal(45, 12, n),
                "income": rng.normal(60000, 20000, n),
            }
        )
        # noisy, low-signal target -> imperfect model with real error
        y = (X["income"] > X["age"] * 800 + rng.normal(0, 40000, n)).astype(int)
        model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
        ctx = TestContext(model=model, X_val=X, y_val=y)
        result = ConfidenceThresholdTest(
            metric="accuracy", threshold=0.99, n_boot=100
        ).run(ctx)
        assert result.status == TestStatus.FAILED
        assert "CI=" in result.detail
        assert "lower CI bound" in result.detail

    def test_upper_bound_mode(self):
        ctx = _make_ctx()
        # model is near-perfect: error rate upper bound should be ~0
        result = ConfidenceThresholdTest(
            metric="accuracy", threshold=2.0, n_boot=100, bound="upper"
        ).run(ctx)
        assert result.status == TestStatus.PASSED

    def test_invalid_bound_errors(self):
        ctx = _make_ctx()
        result = ConfidenceThresholdTest(bound="middle", n_boot=50).run(ctx)
        assert result.status == TestStatus.ERROR
