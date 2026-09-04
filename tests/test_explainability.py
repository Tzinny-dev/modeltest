"""Tests for explainability scenarios."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from modeltest import TestContext
from modeltest.core.base import TestStatus
from modeltest.scenarios import FeatureDominanceTest, TopFeaturesTest
from modeltest.scenarios.explainability import _mean_abs_attribution
from modeltest.wrappers import wrap


def _make_ctx(n=600):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "age": rng.normal(45, 12, n),
            "income": rng.normal(60000, 20000, n),
            "spam": rng.normal(0, 1, n),  # irrelevant feature
        }
    )
    # target depends strongly on income, weakly on age, not at all on spam
    y = (X["income"] > X["age"] * 700 + rng.normal(0, 5000, n)).astype(int)
    model = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)
    return TestContext(model=wrap(model), X_val=X, y_val=y)


class TestMeanAbsAttribution:
    def test_single_matrix(self):
        X = pd.DataFrame({"a": [1], "b": [2]})
        values = np.array([[0.5, -0.3]])
        attr = _mean_abs_attribution(values, X)
        assert attr == {"a": 0.5, "b": 0.3}

    def test_class_list_uses_second_class(self):
        X = pd.DataFrame({"a": [1], "b": [1]})
        values = [np.zeros((1, 2)), np.array([[2.0, 4.0]])]
        attr = _mean_abs_attribution(values, X)
        assert attr == {"a": 2.0, "b": 4.0}


class TestFeatureDominance:
    def test_good_model_passes(self):
        ctx = _make_ctx()
        result = FeatureDominanceTest(max_top_share=0.9).run(ctx)
        assert result.status == TestStatus.PASSED

    def test_single_feature_model_fails(self):
        X = pd.DataFrame({"a": np.random.default_rng(0).normal(size=200)})
        y = (X["a"] > 0).astype(int)
        model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
        ctx = TestContext(model=wrap(model), X_val=X, y_val=y)
        result = FeatureDominanceTest(max_top_share=0.9).run(ctx)
        assert result.status == TestStatus.FAILED


class TestTopFeatures:
    def test_expected_features_present(self):
        ctx = _make_ctx()
        result = TopFeaturesTest(expected_features=["income", "age"], k=2).run(ctx)
        assert result.status == TestStatus.PASSED

    def test_unexpected_top_feature_fails(self):
        ctx = _make_ctx()
        result = TopFeaturesTest(expected_features=["spam"], k=1).run(ctx)
        assert result.status == TestStatus.FAILED
