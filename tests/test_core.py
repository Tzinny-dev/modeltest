"""Pytest suite for the core primitives."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from modeltest import ModelSuite, ModelTest, TestContext
from modeltest.core import base as _base
from modeltest.core.report import to_junit_xml
from modeltest.scenarios import (
    GroupPerformanceTest,
    MinimumAccuracyTest,
    RobustnessTest,
)

TestStatus = _base.TestStatus


def _make_data():
    rng = np.random.default_rng(0)
    n = 3000
    gender = rng.choice(["M", "F"], n)
    X = pd.DataFrame(
        {
            "age": rng.normal(45, 12, n),
            "income": rng.normal(60000, 20000, n),
            "gender_male": (gender == "M").astype(int),
            "_gender": gender,
        }
    )
    y = (X["income"] > X["age"] * 800 + rng.normal(0, 16000, n)).astype(int)
    return X, y


def _build_pass_model(X, y):
    from sklearn.model_selection import train_test_split

    feats = ["age", "income", "gender_male"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.5, random_state=1
    )
    model = RandomForestClassifier(n_estimators=60, random_state=0).fit(
        X_train[feats], y_train
    )
    return model, feats, X_val, y_val


class TestMinimumAccuracy:
    def test_passes_when_above_threshold(self):
        X, y = _make_data()
        model, feats, X_val, y_val = _build_pass_model(X, y)
        result = MinimumAccuracyTest(threshold=0.0).run(
            TestContext(model=model, X_val=X_val[feats], y_val=y_val)
        )
        assert result.passed

    def test_fails_when_below_threshold(self):
        X, y = _make_data()
        model, feats, X_val, y_val = _build_pass_model(X, y)
        result = MinimumAccuracyTest(threshold=0.99).run(
            TestContext(model=model, X_val=X_val[feats], y_val=y_val)
        )
        assert result.status == TestStatus.FAILED


class TestGroupPerformance:
    def test_passes_for_fair_model(self):
        X, y = _make_data()
        model, feats, X_val, y_val = _build_pass_model(X, y)
        result = GroupPerformanceTest(
            metric="accuracy", threshold=0.0, group_col="_gender"
        ).run(TestContext(model=model, X_val=X_val[feats + ["_gender"]], y_val=y_val))
        assert result.passed

    def test_ignores_non_model_columns_in_predict(self):
        X, y = _make_data()
        model, feats, X_val, y_val = _build_pass_model(X, y)
        X_val["_extra"] = np.random.default_rng(0).normal(size=len(X_val))
        result = GroupPerformanceTest(
            metric="accuracy", threshold=0.0, group_col="_gender"
        ).run(
            TestContext(
                model=model,
                X_val=X_val[feats + ["_gender", "_extra"]],
                y_val=y_val,
            )
        )
        assert result.passed


class TestRobustness:
    def test_passes_small_noise(self):
        X, y = _make_data()
        model, feats, X_val, y_val = _build_pass_model(X, y)
        result = RobustnessTest(noise_std=10, max_drop=0.05, metric="accuracy").run(
            TestContext(model=model, X_val=X_val[feats], y_val=y_val)
        )
        assert result.passed


class TestSuite:
    def test_aggregates_results(self):
        X, y = _make_data()
        model, feats, X_val, y_val = _build_pass_model(X, y)
        suite = ModelSuite(name="t")
        suite.add_tests(
            MinimumAccuracyTest(threshold=0.0),
            MinimumAccuracyTest(threshold=0.99),  # will fail
        )
        result = suite.run(model, X_val[feats], y_val)
        assert result.num_passed == 1
        assert result.num_failed == 1
        assert not result.passed

    def test_custom_test_subclass(self):
        class AlwaysPass(ModelTest):
            def test(self, ctx):
                return True

        X, y = _make_data()
        model, feats, X_val, y_val = _build_pass_model(X, y)
        suite = ModelSuite(name="t")
        suite.add_test(AlwaysPass())
        assert suite.run(model, X_val[feats], y_val).passed


class TestReport:
    def test_junit_xml_is_well_formed(self):
        X, y = _make_data()
        model, feats, X_val, y_val = _build_pass_model(X, y)
        suite = ModelSuite(name="fraud")
        suite.add_test(MinimumAccuracyTest(threshold=0.0))
        result = suite.run(model, X_val[feats], y_val)
        xml = to_junit_xml(result)
        assert "<testsuite" in xml
        assert 'name="fraud"' in xml
        assert "MinimumAccuracyTest" in xml


class TestFingerprintFallbacks:
    def test_pickle_fallback_for_plain_array(self):
        ctx = TestContext(model=None, X_val=np.array([[1.0, 2.0]]), y_val=[1])
        fp1 = ctx._fingerprint(np.array([[1.0, 2.0]]))
        fp2 = ctx._fingerprint(np.array([[1.0, 2.0]]))
        assert fp1 == fp2
        assert ctx._fingerprint(np.array([[9.0, 9.0]])) != fp1

    def test_id_fallback_when_unpicklable(self, monkeypatch):
        ctx = TestContext(model=None, X_val=[], y_val=[])
        import pickle as _pickle

        def _boom(*a, **k):
            raise RuntimeError("cannot pickle")

        monkeypatch.setattr(_pickle, "dumps", _boom)
        x = object()
        key = ctx._fingerprint([x])  # list holds an object -> id fallback
        assert key.startswith("id-")

    def test_predict_proba_through_context(self):
        X, y = _make_data()
        model, feats, X_val, y_val = _build_pass_model(X, y)
        ctx = TestContext(model=model, X_val=X_val[feats], y_val=y_val)
        proba = ctx.predict_proba()
        assert proba is not None
        assert proba.shape == (len(X_val), 2)

    def test_predict_none_uses_xval(self):
        X, y = _make_data()
        model, feats, X_val, y_val = _build_pass_model(X, y)
        ctx = TestContext(model=model, X_val=X_val[feats], y_val=y_val)
        pred = ctx.predict()  # X=None -> predicts on X_val
        assert pred.shape == (len(X_val),)

    def test_cache_disabled_predicts_fresh(self, monkeypatch):
        X, y = _make_data()
        model, feats, X_val, y_val = _build_pass_model(X, y)
        calls = []
        from modeltest.wrappers import wrap

        wrapped = wrap(model)
        original_predict = wrapped.predict

        def counting_predict(X):
            calls.append(X)
            return original_predict(X)

        monkeypatch.setattr(wrapped, "predict", counting_predict)
        ctx = TestContext(
            model=wrapped, X_val=X_val[feats], y_val=y_val, cache_predictions=False
        )
        ctx._wrapper = wrapped
        ctx.predict()
        ctx.predict()
        assert len(calls) == 2
