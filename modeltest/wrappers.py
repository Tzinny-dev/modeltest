"""Model adapters.

``TestContext`` talks to models through a small common interface:

    predict(X)        -> class labels
    predict_proba(X)  -> probability estimates (optional)

``wrap(model)`` returns a normalized object implementing this interface for the
given model, dispatching on the framework:

- scikit-learn / numpy (BaseEstimator)
- PyTorch (torch.nn.Module) — lazily imported
- Keras / TensorFlow — lazily imported

If the model can't be recognised, we assume it already follows the common
interface (e.g. a user-provided wrapper) and let ``predict``/``predict_proba``
resolve dynamically.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np


class ModelWrapper:
    """Normalized interface all scenarios use to query a model."""

    __test__ = False  # don't let pytest collect this as a test class

    def __init__(self, model: Any):
        self.model = model
        self._predict_fn: Optional[Callable] = None
        self._proba_fn: Optional[Callable] = None

    def predict(self, X: Any) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: Any) -> Optional[np.ndarray]:
        raise NotImplementedError

    @property
    def feature_names_in_(self) -> Optional[list]:
        """Expose the model's feature names when available (for filtering)."""
        return getattr(self.model, "feature_names_in_", None)


class SklearnModel(ModelWrapper):
    """Adapter for scikit-learn style estimators exposing ``predict``."""

    def predict(self, X: Any) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> Optional[np.ndarray]:
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(X))
        return None


class SklearnClassifier(SklearnModel):
    """Adapter for classifiers: labels + probabilities."""

    def predict_proba(self, X: Any) -> Optional[np.ndarray]:
        return np.asarray(self.model.predict_proba(X))


class TorchModel(ModelWrapper):
    """Adapter for a PyTorch ``nn.Module`` trained for classification.

    ``predict`` returns argmax class indices; ``predict_proba`` returns
    softmax probabilities. Expects ``X`` as a numpy array or torch tensor.
    """

    def __init__(self, model: Any, *, input_key: str = "images", device: Any = None):
        super().__init__(model)
        self.input_key = input_key
        self.device = device

    @staticmethod
    def _to_tensor(X: Any, torch: Any):
        if isinstance(X, np.ndarray):
            return torch.as_tensor(X)
        if X is not None and not isinstance(X, torch.Tensor):
            return torch.as_tensor(X)
        return X

    def _forward(self, X: Any):
        try:
            import torch
        except ImportError:
            torch = None

        if torch is not None:
            if self.model.training:
                self.model.eval()
            with torch.no_grad():
                src = self._to_tensor(X, torch)
                if self.device is not None:
                    src = src.to(self.device)
                return self.model(src)

        # No real torch available (e.g. a stand-in in tests): pass data through.
        if self.model.training:
            self.model.eval()
        if isinstance(X, np.ndarray):
            return self.model(X)
        return self.model(X)

    def predict(self, X: Any) -> np.ndarray:
        out = self._forward(X)
        if hasattr(out, "detach"):
            data = out.detach().cpu().numpy()
        else:
            data = np.asarray(out).squeeze()
        return np.asarray(data).argmax(axis=1).astype(int)

    def predict_proba(self, X: Any) -> Optional[np.ndarray]:
        out = self._forward(X)
        data = out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
        data = np.asarray(data)
        if data.ndim == 1:
            data = data[:, None]
        exp = np.exp(data - data.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)


class KerasModel(ModelWrapper):
    """Adapter for a compiled Keras/TensorFlow model."""

    def __init__(self, model: Any, *, multiclass: bool = False):
        super().__init__(model)
        self.multiclass = multiclass

    def predict(self, X: Any) -> np.ndarray:
        proba = np.asarray(self.model.predict(X, verbose=0))
        if self.multiclass or proba.ndim == 2 and proba.shape[1] > 2:
            return proba.argmax(axis=1)
        return (proba > 0.5).astype(int).ravel()

    def predict_proba(self, X: Any) -> Optional[np.ndarray]:
        return np.asarray(self.model.predict(X, verbose=0))


def wrap(model: Any, **kwargs: Any) -> ModelWrapper:
    """Return a normalized :class:`ModelWrapper` for ``model``.

    ``kwargs`` are forwarded to the adapter (e.g. ``input_key`` for Torch,
    ``multiclass`` for Keras). If the model is already a ``ModelWrapper`` it is
    returned unchanged.
    """
    if isinstance(model, ModelWrapper):
        return model

    # Torch
    try:
        import torch  # noqa: F401

        if isinstance(model, torch.nn.Module):
            return TorchModel(model, **kwargs)
    except ImportError:
        pass

    # Keras / TF
    try:
        import tensorflow as tf  # noqa: F401

        if hasattr(tf, "keras") and isinstance(model, tf.keras.Model):
            return KerasModel(model, **kwargs)
    except ImportError:
        pass

    # scikit-learn / sklearn-style
    try:
        from sklearn.base import BaseEstimator

        if isinstance(model, BaseEstimator):
            if hasattr(model, "predict_proba"):
                return SklearnClassifier(model)
            return SklearnModel(model)
    except ImportError:
        pass

    # Fallback: optimistically assume the common interface.
    return SklearnModel(model)
