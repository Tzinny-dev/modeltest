"""Drift tests: detect distribution shift between train and validation data.

These tests compare the *data* distributions (not the model), so they don't
need the model to make predictions — only the two datasets. That is why
``test`` reads from ``ctx.X_train`` / ``ctx.X_val`` rather than using the model.
"""

from __future__ import annotations

from typing import Any, List, Optional

from modeltest.core.base import ModelTest, TestContext
from modeltest.scenarios._utils import psi


class DataDriftTest(ModelTest):
    """Assert Population Stability Index stays below a threshold per column.

    ``expected`` is the reference distribution (typically training), ``actual``
    is the current one (typically validation / production). By default
    ``expected`` and ``actual`` come from the context's ``X_train`` and
    ``X_val``; you can override them.
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        max_psi: float = 0.25,
        n_bins: int = 10,
    ):
        """Configure the PSI drift check.

        Args:
            feature_cols: Columns to check; ``None`` checks all numeric
                columns of ``X_train``.
            max_psi: Maximum tolerated PSI per column.
            n_bins: Quantile bins used to discretize each column.
        """
        self.feature_cols = feature_cols
        self.max_psi = max_psi
        self.n_bins = n_bins

    def test(self, ctx: TestContext) -> Any:
        """Compare train vs. validation PSI per column.

        Raises:
            ValueError: If the context has no ``X_train``.
            AssertionError: If a column is missing from ``X_val`` or any
                PSI exceeds ``max_psi``.
        """
        if ctx.X_train is None:
            raise ValueError("DataDriftTest requires X_train in the context")

        cols = self.feature_cols or _numeric_cols(ctx.X_train)
        for col in cols:
            if col not in ctx.X_val.columns:
                raise AssertionError(f"Column {col!r} missing from validation data")
            p = psi(ctx.X_train[col], ctx.X_val[col], n_bins=self.n_bins)
            assert p <= self.max_psi, (
                f"PSI for {col} = {p:.4f} exceeds max_psi {self.max_psi}"
            )


class KSTest(ModelTest):
    """Assert the two-sample KS p-value stays above a threshold per column.

    A low p-value indicates the train and validation distributions for a column
    are significantly different (drift).
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        min_p_value: float = 0.05,
    ):
        """Configure the KS drift check.

        Args:
            feature_cols: Columns to check; ``None`` checks all numeric
                columns of ``X_train``.
            min_p_value: Minimum tolerated KS p-value per column.
        """
        self.feature_cols = feature_cols
        self.min_p_value = min_p_value

    def test(self, ctx: TestContext) -> Any:
        """Run the two-sample KS test per column against the context data.

        Raises:
            ValueError: If the context has no ``X_train``.
            AssertionError: If a column is missing from ``X_val`` or any
                p-value falls below ``min_p_value``.
        """
        if ctx.X_train is None:
            raise ValueError("KSTest requires X_train in the context")

        from scipy import stats

        cols = self.feature_cols or _numeric_cols(ctx.X_train)
        for col in cols:
            if col not in ctx.X_val.columns:
                raise AssertionError(f"Column {col!r} missing from validation data")
            _, p = stats.ks_2samp(ctx.X_train[col], ctx.X_val[col])
            assert p >= self.min_p_value, (
                f"KS p-value for {col} = {p:.4g} < min_p_value {self.min_p_value}"
            )


def _numeric_cols(df: Any) -> List[str]:
    try:
        return list(df.select_dtypes(include="number").columns)
    except AttributeError:
        return list(df.columns)
