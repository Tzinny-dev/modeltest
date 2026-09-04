"""Tests for fairness scenarios."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from modeltest import TestContext
from modeltest.core.base import TestStatus
from modeltest.scenarios import EqualOpportunityTest, StatisticalParityTest
from modeltest.scenarios._utils import group_rates


def _biased_model():
    """A model that systematically under-serves group 'B'.

    Feature values are strongly correlated with group membership, so the model
    gives group B far lower scores/higher negatives -> clear selection-parity
    and TPR disparity between A and B.
    """
    rng = np.random.default_rng(5)
    n = 6000
    group = rng.choice(["A", "B"], n)
    feat = np.where(group == "A", rng.normal(1.0, 0.8, n), rng.normal(-1.0, 0.8, n))
    y = (feat + rng.normal(0, 0.4, n) > 0).astype(int)
    X = pd.DataFrame({"feat": feat})

    idx = np.arange(n)
    train_idx, val_idx = train_test_split(idx, test_size=0.5, random_state=0)
    model = RandomForestClassifier(n_estimators=60, random_state=0).fit(
        X.loc[train_idx], y[train_idx]
    )
    return model, X.loc[val_idx], y[val_idx], group[val_idx]


def _fair_model():
    """A model with no group-driven selection difference."""
    rng = np.random.default_rng(2)
    n = 6000
    group = rng.choice(["A", "B"], n)
    feat = rng.normal(0, 1, n)  # feature independent of group
    y = (feat + rng.normal(0, 0.3, n) > 0).astype(int)
    X = pd.DataFrame({"feat": feat})

    idx = np.arange(n)
    train_idx, val_idx = train_test_split(idx, test_size=0.5, random_state=0)
    model = RandomForestClassifier(n_estimators=60, random_state=0).fit(
        X.loc[train_idx], y[train_idx]
    )
    return model, X.loc[val_idx], y[val_idx], group[val_idx]


def _ctx(model, X_val, y_val, group_labels):
    X = X_val.copy()
    X["group"] = group_labels
    return TestContext(model=model, X_val=X, y_val=y_val)


class TestGroupRates:
    def test_group_rates_compute(self):
        yt = np.array([1, 0, 1, 1, 0, 1])
        yp = np.array([1, 0, 0, 1, 0, 1])
        g = np.array(["a", "a", "a", "b", "b", "b"])
        rates = group_rates(yt, yp, g)
        assert set(rates) == {"a", "b"}
        assert "selection_rate" in rates["a"] and "tpr" in rates["a"]


class TestEqualOpportunity:
    def test_fair_model_passes(self):
        model, X_val, y_val, group = _fair_model()
        result = EqualOpportunityTest("group", max_diff=0.1).run(
            _ctx(model, X_val, y_val, group)
        )
        assert result.status == TestStatus.PASSED

    def test_biased_model_fails(self):
        model, X_val, y_val, group = _biased_model()
        result = EqualOpportunityTest("group", max_diff=0.05).run(
            _ctx(model, X_val, y_val, group)
        )
        assert result.status == TestStatus.FAILED


class TestStatisticalParity:
    def test_fair_model_passes(self):
        model, X_val, y_val, group = _fair_model()
        result = StatisticalParityTest("group", max_diff=0.1).run(
            _ctx(model, X_val, y_val, group)
        )
        assert result.status == TestStatus.PASSED

    def test_biased_model_fails_difference(self):
        model, X_val, y_val, group = _biased_model()
        result = StatisticalParityTest("group", max_diff=0.05, min_ratio=0.5).run(
            _ctx(model, X_val, y_val, group)
        )
        assert result.status == TestStatus.FAILED
