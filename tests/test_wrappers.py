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

    def test_wrap_sklearn_regressor_without_proba(self):
        from sklearn.ensemble import RandomForestRegressor

        Xtr, Xv, ytr, _ = _dataset()
        model = RandomForestRegressor(n_estimators=10, random_state=0).fit(Xtr, ytr)
        w = wrap(model)
        assert isinstance(w, SklearnModel)  # BaseEstimator without predict_proba
        assert w.predict_proba(Xv) is None

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

    def test_no_torch_fallback_without_detach(self):
        # Input is a plain list (not ndarray) and the model's output has no
        # .detach -> both the 108 (list branch) and 115 (squeeze) paths run.
        class SimpleModule:
            training = False

            def __call__(self, x):
                import numpy as _np

                return _np.asarray(x) @ _np.array([[1.0, -1.0], [-1.0, 1.0]]).T

            def eval(self):
                self.training = False

        a = TorchModel(SimpleModule())
        pred = a.predict([[1.0, -1.0], [-1.0, 1.0]])  # batch of 2
        np.testing.assert_array_equal(pred, [0, 1])

    def test_predict_proba_1d_output_reshaped(self):
        # predict_proba on a single-class output -> data[:, None] reshape path.
        class SingleLogit:
            training = False

            def __call__(self, x):
                return np.asarray(x).sum(axis=1, keepdims=False)

            def eval(self):
                self.training = False

        a = TorchModel(SingleLogit())
        proba = a.predict_proba(np.array([[1.0, 2.0], [3.0, 4.0]]))
        # softmax over a 1-column class logit -> all 1.0
        assert proba.shape == (2, 1)
        np.testing.assert_allclose(proba, 1.0)

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


class TestKerasDispatchViaWrap:
    def test_wrap_dispatches_to_keras(self, monkeypatch):
        import sys
        import types

        tf_mod = types.ModuleType("tensorflow")

        class FakeKeras:
            class Model:
                pass

        tf_mod.keras = FakeKeras
        monkeypatch.setitem(sys.modules, "tensorflow", tf_mod)

        class FakeTFModel(tf_mod.keras.Model):
            def predict(self, X, verbose=0):
                return np.zeros((len(X), 1))

        w = wrap(FakeTFModel())
        assert isinstance(w, KerasModel)

        pred = w.predict(np.zeros((4, 3)))
        np.testing.assert_array_equal(pred, np.zeros(4))


class TestWrapImportErrorGuards:
    def test_sklearn_import_error_uses_fallback(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _no_sklearn_base(name, *args, **kwargs):
            if name == "sklearn.base":
                raise ImportError("simulated missing sklearn")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_sklearn_base)

        class Generic:
            def predict(self, X):
                return np.zeros(len(X), dtype=int)

        w = wrap(Generic())  # sklearn import fails -> generic SklearnModel fallback
        assert isinstance(w, SklearnModel)
        assert w.predict(np.zeros(3)).shape == (3,)


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

    def test_sklearn_modelwith_proba_returns_array(self):
        # SklearnModel (not the classifier subclass) still honours predict_proba.
        from sklearn.linear_model import LogisticRegression

        Xtr, Xv, ytr, _ = _dataset()
        model = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
        w = SklearnModel(model)
        proba = w.predict_proba(Xv)
        assert proba.shape == (len(Xv), 2)

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


class _FakeTorchTensor:
    def __init__(self, data):
        self._data = np.asarray(data)

    def numpy(self):
        return self._data

    def cpu(self):
        return self

    def detach(self):
        return self

    def to(self, device):
        return self


class _TorchNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _install_fake_torch(monkeypatch):
    """Register a minimal torch stub so wrap() can dispatch to TorchModel."""
    import sys
    import types

    torch_mod = types.ModuleType("torch")

    class FakeNN:
        class Module:
            pass

    torch_mod.nn = FakeNN
    torch_mod.Tensor = _FakeTorchTensor
    torch_mod.as_tensor = lambda x: _FakeTorchTensor(x)
    torch_mod.no_grad = lambda: _TorchNoGrad()
    torch_mod.device = lambda *a, **k: "cpu"
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    return torch_mod


class TestTorchDispatchViaWrap:
    def test_wrap_dispatches_to_torch(self, monkeypatch):
        torch_mod = _install_fake_torch(monkeypatch)
        from modeltest.wrappers import TorchModel

        class FakeModule(torch_mod.nn.Module):
            training = False
            W = np.array([[1.0, -1.0], [-1.0, 1.0]])

            def __call__(self, x):
                data = getattr(x, "numpy", lambda: x)()
                logits = np.asarray(data) @ self.W.T
                return logits

            def eval(self):
                self.training = False

        w = wrap(FakeModule())
        assert isinstance(w, TorchModel)

    def test_torch_forward_real_path(self, monkeypatch):
        _install_fake_torch(monkeypatch)

        class FakeModule:
            training = True
            W = np.array([[1.0, -1.0], [-1.0, 1.0]])

            def __call__(self, x):
                data = np.asarray(x.numpy())
                return _FakeTorchTensor(data @ self.W.T)

            def eval(self):
                self.training = False

        a = TorchModel(FakeModule())
        X = np.array([[1.0, -1.0], [-1.0, 1.0]])
        pred = a.predict(X)  # torch.no_grad + eval path
        np.testing.assert_array_equal(pred, [0, 1])
        proba = a.predict_proba(X)
        assert proba.shape == (2, 2)

    def test_torch_to_tensor_with_tensor_input(self, monkeypatch):
        _install_fake_torch(monkeypatch)

        class FakeModule:
            training = False
            W = np.array([[1.0, -1.0], [-1.0, 1.0]])

            def __call__(self, t):
                # _to_tensor already returned an _FakeTorchTensor
                data = t.numpy()
                return _FakeTorchTensor(data @ self.W.T)

            def eval(self):
                self.training = False

        a = TorchModel(FakeModule())
        pred = a.predict(_FakeTorchTensor([[1.0, -1.0], [-1.0, 1.0]]))
        np.testing.assert_array_equal(pred, [0, 1])

    def test_torch_forward_with_device(self, monkeypatch):
        # device != None -> src.to(device) path (line 100).
        _install_fake_torch(monkeypatch)

        class FakeModule:
            training = False
            W = np.array([[1.0, -1.0], [-1.0, 1.0]])

            def __call__(self, t):
                data = t.numpy()
                return _FakeTorchTensor(data @ self.W.T)

            def eval(self):
                self.training = False

        a = TorchModel(FakeModule(), device="cuda:0")
        pred = a.predict(np.array([[1.0, -1.0], [-1.0, 1.0]]))
        np.testing.assert_array_equal(pred, [0, 1])

    def test_torch_to_tensor_list_input(self, monkeypatch):
        # A plain list is neither ndarray nor tensor -> torch.as_tensor branch
        # (line 85).
        _install_fake_torch(monkeypatch)

        class FakeModule:
            training = False
            W = np.array([[1.0, -1.0], [-1.0, 1.0]])

            def __call__(self, t):
                data = t.numpy()
                return _FakeTorchTensor(data @ self.W.T)

            def eval(self):
                self.training = False

        a = TorchModel(FakeModule())
        pred = a.predict([[1.0, -1.0], [-1.0, 1.0]])  # list input
        np.testing.assert_array_equal(pred, [0, 1])
