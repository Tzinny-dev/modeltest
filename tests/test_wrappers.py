"""Tests for model wrappers / framework adapters."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from modeltest import TestContext
from modeltest.scenarios import MinimumAccuracyTest
from modeltest.wrappers import (
    KerasModel,
    ModelWrapper,
    SklearnClassifier,
    SklearnModel,
    TorchModel,
    wrap,
)


def _dataset():
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(0)
    n = 800
    X = pd.DataFrame(
        {"age": rng.normal(45, 12, n), "income": rng.normal(60000, 20000, n)}
    )
    y = (X["income"] > X["age"] * 800 + rng.normal(0, 16000, n)).astype(int)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.4, random_state=0
    )
    return X_train, X_val, y_train, y_val


class TestSklearnWrapper:
    def test_wrap_sklearn_classifier_provides_proba(self):
        Xtr, Xv, ytr, yv = _dataset()
        model = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
        w = wrap(model)
        assert isinstance(w, SklearnClassifier)
        pred = w.predict(Xv)
        proba = w.predict_proba(Xv)
        assert pred.shape == yv.shape
        assert proba.shape == (len(Xv), 2)

    def test_context_uses_wrapper_for_classes(self):
        Xtr, Xv, ytr, yv = _dataset()
        model = RandomForestClassifier(n_estimators=20, random_state=0).fit(Xtr, ytr)
        ctx = TestContext(model=model, X_val=Xv, y_val=yv)
        result = MinimumAccuracyTest(threshold=0.5).run(ctx)
        assert result.passed
        assert ctx._wrapped() is not None

    def test_wrap_passthrough_for_existing_wrapper(self):
        Xtr, Xv, ytr, yv = _dataset()
        model = RandomForestClassifier(n_estimators=20, random_state=0).fit(Xtr, ytr)
        w = wrap(model)
        assert wrap(w) is w


class _FakeTorchTensor:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)


class _FakeTorchOut:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class _FakeTorchModule:
    """Minimal stand-in mimicking torch.nn.Module for adapter testing."""

    training = False

    def __init__(self, weights):
        self.weights = weights
        self._logits = self.weights @ np.eye(weights.shape[1])  # always 1

    def __call__(self, x):
        x = np.asarray(x.data)
        logits = x @ self.weights.T
        return _FakeTorchOut(logits)

    def eval(self):
        self.training = False
        return self


class TestTorchWrapper:
    def test_torch_adapter_predict_and_proba(self):
        # 2 features -> 2 classes, weights rows are class scores
        weights = np.array([[1.0, -1.0], [-1.0, 1.0]])
        m = _FakeTorchModule(weights)
        a = TorchModel(m)
        X = np.array([[1.0, -1.0], [-1.0, 1.0]])
        pred = a.predict(X)
        proba = a.predict_proba(X)
        assert pred.shape == (2,)
        # Class 0 wins for [1,-1], class 1 for [-1,1]
        assert pred[0] == 0 and pred[1] == 1
        assert proba.shape == (2, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_no_torch_available_fallback(self):
        # Without torch installed, _forward passes ndarray straight to the model.
        weights = np.array([[1.0, -1.0], [-1.0, 1.0]])
        m = _FakeTorchModule(weights)
        a = TorchModel(m)
        X = np.array([[1.0, -1.0]])
        pred = a.predict(X)
        assert pred[0] == 0

    def test_predict_proba_1d_input_reshaped(self):
        # Trigger the data[:, None] path via a single-row input squeezed to 1D.
        m = _FakeTorchModule(np.array([[1.0, -1.0], [-1.0, 1.0]]))
        a = TorchModel(m)
        proba = a.predict_proba(np.array([[1.0, -1.0]]))
        assert proba.shape == (1, 2)

    def test_eval_called_when_training(self):
        weights = np.array([[1.0, -1.0], [-1.0, 1.0]])
        m = _FakeTorchModule(weights)
        m.training = True
        a = TorchModel(m)
        a.predict(np.array([[1.0, -1.0]]))
        assert m.training is False


class _FakeKerasModel:
    """Minimal stand-in mimicking a compiled Keras model."""

    def __init__(self, proba):
        self._proba = proba

    def predict(self, X, verbose=0):
        return self._proba


class TestKerasWrapper:
    def test_binary_threshold_decision(self):
        m = _FakeKerasModel(np.array([[0.9], [0.2], [0.6]]))
        a = KerasModel(m)
        pred = a.predict(np.zeros((3, 1)))
        np.testing.assert_array_equal(pred, np.array([1, 0, 1]))
        proba = a.predict_proba(np.zeros((3, 1)))
        assert proba.shape == (3, 1)

    def test_multiclass_argmax(self):
        proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
        m = _FakeKerasModel(proba)
        a = KerasModel(m, multiclass=True)
        np.testing.assert_array_equal(a.predict(np.zeros((2, 3))), np.array([0, 2]))


class TestFallbackDispatch:
    def test_unknown_model_treated_as_sklearn(self):
        class Generic:
            def predict(self, X):
                return np.zeros(len(X), dtype=int)

        w = wrap(Generic())
        assert isinstance(w, ModelWrapper)
        assert w.predict(np.zeros(5)).shape == (5,)


class TestModelWrapperBase:
    def test_predict_raises_not_implemented(self):
        class Bare(ModelWrapper):
            pass

        with pytest.raises(NotImplementedError):
            Bare(None).predict(np.zeros(3))

    def test_predict_proba_raises_not_implemented(self):
        class Bare(ModelWrapper):
            pass

        with pytest.raises(NotImplementedError):
            Bare(None).predict_proba(np.zeros(3))

    def test_sklearn_model_without_proba_returns_none(self):
        class Regressor:
            def predict(self, X):
                return np.zeros(len(X))

        w = SklearnModel(Regressor())
        assert w.predict_proba(np.zeros(3)) is None

    def test_feature_names_in_exposed(self):
        class WithNames:
            feature_names_in_ = ["a", "b"]

        w = wrap(WithNames())
        # WithNames is not a BaseEstimator, so it falls to generic SklearnModel
        assert w.feature_names_in_ == ["a", "b"]

    def test_feature_names_none_when_absent(self):
        class NoNames:
            def predict(self, X):
                return np.zeros(len(X))

        w = wrap(NoNames())
        assert w.feature_names_in_ is None
