"""Robustness tests: verify performance holds under input perturbation."""

from __future__ import annotations

from typing import Any

import numpy as np

from modeltest.core.base import ModelTest, TestContext
from modeltest.scenarios._utils import model_features
from modeltest.scenarios.performance import resolve_metric


class RobustnessTest(ModelTest):
    """Assert adding Gaussian noise to numeric features does not degrade
    the chosen metric by more than ``max_drop`` relative to the clean data.
    """

    def __init__(
        self,
        noise_std: float = 0.05,
        max_drop: float = 0.05,
        metric: str = "accuracy",
        seed: int = 42,
    ):
        self.noise_std = noise_std
        self.max_drop = max_drop
        self.metric = metric
        self.seed = seed

    def test(self, ctx: TestContext) -> Any:
        fn = resolve_metric(self.metric)
        X = ctx.X_val
        X_model = model_features(ctx.model, X)
        rng = np.random.default_rng(self.seed)

        X_noisy = X_model.copy()
        numeric_cols = X_model.select_dtypes(include=[np.number]).columns
        noise = rng.normal(0, self.noise_std, size=X_model[numeric_cols].shape)
        X_noisy[numeric_cols] = X_model[numeric_cols].to_numpy() + noise

        y_pred_clean = ctx.predict(X_model)
        y_pred_noisy = ctx.predict(X_noisy)

        score_clean = fn(ctx.y_val, y_pred_clean)
        score_noisy = fn(ctx.y_val, y_pred_noisy)
        drop = score_clean - score_noisy
        assert drop <= self.max_drop, (
            f"metric dropped by {drop:.4f} "
            f"(clean {score_clean:.4f} -> noisy {score_noisy:.4f}) "
            f"> max_drop {self.max_drop}"
        )
