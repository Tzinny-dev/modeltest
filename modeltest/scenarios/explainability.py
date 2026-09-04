"""Explainability tests.

These scenarios measure how the model uses its input features — coherence of
the top features and stability of the attribution. Explainer backends are
plugged in lazily so that heavy dependencies (SHAP) are only required when the
scenario actually runs, and users can pass their own explainer callable.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import numpy as np

from modeltest.core.base import ModelTest, TestContext
from modeltest.scenarios._utils import model_features

Explainer = Callable[[Any, Any], Any]
"""Signature: explainer(model, X) -> object exposing .values (n_samples, n_features)."""


def _default_shap_explainer(model: Any, X: Any) -> Any:
    """Build a SHAP explainer appropriate for the wrapped model."""
    import shap

    base = getattr(model, "model", model)  # unwrap our ModelWrapper
    if hasattr(base, "predict_proba") or hasattr(base, "predict"):
        try:
            # TreeExplainer works for tree ensembles and logistic regression.
            explainer = shap.TreeExplainer(base)
            return explainer.shap_values(X, check_additivity=False)
        except Exception:  # noqa: BLE001 - fall back to KernelExplainer
            pass

    explainer = shap.KernelExplainer(base.predict, X[: min(100, len(X))])
    return explainer.shap_values(X)


def _mean_abs_attribution(values: Any, X: Any) -> "dict[str, float]":
    """Mean absolute SHAP value per feature column."""
    arr = np.asarray(values)
    # For classifiers SHAP may return a list (one matrix per class).
    if arr.ndim == 3:
        arr = arr[1] if arr.shape[0] > 1 else arr[0]
    arr = np.atleast_2d(arr)
    means = np.abs(arr).mean(axis=0)
    cols = (
        list(X.columns)
        if hasattr(X, "columns")
        else [f"f{i}" for i in range(arr.shape[1])]
    )
    return dict(zip(cols, map(float, means)))


class FeatureDominanceTest(ModelTest):
    """Assert the model does not rely on a single feature to the exclusion of all.

    Computes mean abs SHAP attribution per feature and checks that the top
    feature's share stays below ``max_top_share`` (default 0.9). This catches
    degenerate models that effectively use only one column.
    """

    def __init__(
        self,
        explainer: Optional[Explainer] = None,
        max_top_share: float = 0.9,
    ):
        self.explainer = explainer or _default_shap_explainer
        self.max_top_share = max_top_share

    def test(self, ctx: TestContext) -> Any:
        X = model_features(ctx._wrapped(), ctx.X_val)
        values = self.explainer(ctx._wrapped(), X)
        attr = _mean_abs_attribution(values, X)
        total = sum(attr.values())
        if total <= 0:
            raise AssertionError("All SHAP attributions are zero; model is degenerate")
        top = max(attr.values())
        share = top / total
        assert share <= self.max_top_share, (
            f"Top feature accounts for {share:.2f} of attribution "
            f"(> max_top_share {self.max_top_share}); attributions: {attr}"
        )


class TopFeaturesTest(ModelTest):
    """Assert the top-K features by attribution are within the expected set.

    ``expected_features`` is the set the team considers plausible/desired.
    Fails if any of the top ``k`` features falls outside it.
    """

    def __init__(
        self,
        expected_features: List[str],
        k: int = 3,
        explainer: Optional[Explainer] = None,
    ):
        self.expected_features = set(expected_features)
        self.k = k
        self.explainer = explainer or _default_shap_explainer

    def test(self, ctx: TestContext) -> Any:
        X = model_features(ctx._wrapped(), ctx.X_val)
        values = self.explainer(ctx._wrapped(), X)
        attr = _mean_abs_attribution(values, X)
        ranked = sorted(attr, key=attr.get, reverse=True)[: self.k]
        unexpected = [f for f in ranked if f not in self.expected_features]
        assert not unexpected, (
            f"Unexpected top-{self.k} features: {unexpected}; attributions: {attr}"
        )
