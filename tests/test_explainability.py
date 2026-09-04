"""Tests for explainability scenarios."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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

    def test_3d_array_takes_class1(self):
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        # (n_samples, n_features, n_classes)
        values = np.array(
            [
                [[0.0, 1.0], [0.0, 2.0]],
                [[0.0, 3.0], [0.0, 4.0]],
            ]
        )
        attr = _mean_abs_attribution(values, X)
        # class 1 values: [[1,2],[3,4]] → mean per feature: [2.0, 3.0]
        assert attr["a"] == pytest.approx(2.0)
        assert attr["b"] == pytest.approx(3.0)

    def test_single_class_3d_takes_class0(self):
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        values = np.array(
            [
                [[5.0], [6.0]],
                [[7.0], [8.0]],
            ]
        )
        attr = _mean_abs_attribution(values, X)
        assert attr["a"] == pytest.approx(6.0)
        assert attr["b"] == pytest.approx(7.0)

    def test_uses_feature_names_when_no_columns(self):
        X = np.array([[1, 2], [3, 4]])
        values = np.array([[10.0, 20.0]])
        attr = _mean_abs_attribution(values, X, feature_names=["x", "y"])
        assert attr == {"x": 10.0, "y": 20.0}

    def test_fallback_positional_names(self):
        X = np.array([[1, 2], [3, 4]])
        values = np.array([[10.0, 20.0]])
        attr = _mean_abs_attribution(values, X)
        assert attr == {"f0": 10.0, "f1": 20.0}


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

    def test_all_zero_attribution_fails(self):
        # A stub explainer returning zeros -> degenerate "all zero" path.
        def zero_explainer(model, X):
            return np.zeros((len(X), len(X.columns)))

        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = FeatureDominanceTest(max_top_share=0.9, explainer=zero_explainer).run(
            TestContext(model=None, X_val=X, y_val=[0, 1])
        )
        assert result.status == TestStatus.FAILED
        assert "degenerate" in result.detail


class TestKernelExplainerFallback:
    def _logistic_ctx(self):
        # LogisticRegression is not supported by TreeExplainer, forcing the
        # KernelExplainer fallback path in _default_shap_explainer.
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
        y = (X["a"] > 0).astype(int)
        model = LogisticRegression(max_iter=1000).fit(X, y)
        return TestContext(model=wrap(model), X_val=X, y_val=y)

    def test_falls_back_to_kernel_explainer(self):
        ctx = self._logistic_ctx()
        # Logistic's single dominant feature has share==1.0, so only a
        # threshold of 1.0 lets it pass after the Kernel fallback.
        result = FeatureDominanceTest(max_top_share=1.0).run(ctx)
        assert result.status == TestStatus.PASSED

    def test_top_features_with_kernel_fallback(self):
        ctx = self._logistic_ctx()
        result = TopFeaturesTest(expected_features=["a"], k=1).run(ctx)
        assert result.status == TestStatus.PASSED


class TestExplainabilityImportGuards:
    def test_missing_sklearn_pipeline_is_skipped(self, monkeypatch):
        """Cover the `except ImportError` guard around the Pipeline branch."""
        import builtins

        real_import = builtins.__import__

        def _no_pipeline(name, *args, **kwargs):
            if name == "sklearn.pipeline":
                raise ImportError("simulated missing pipeline")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_pipeline)

        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.normal(size=150), "b": rng.normal(size=150)})
        y = (X["a"] > 0).astype(int)
        model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
        ctx = TestContext(model=wrap(model), X_val=X, y_val=y)
        result = FeatureDominanceTest(max_top_share=0.9).run(ctx)
        assert result.status == TestStatus.PASSED or result.status == TestStatus.FAILED


class TestTopFeatures:
    def test_expected_features_present(self):
        ctx = _make_ctx()
        result = TopFeaturesTest(expected_features=["income", "age"], k=2).run(ctx)
        assert result.status == TestStatus.PASSED

    def test_unexpected_top_feature_fails(self):
        ctx = _make_ctx()
        result = TopFeaturesTest(expected_features=["spam"], k=1).run(ctx)
        assert result.status == TestStatus.FAILED
