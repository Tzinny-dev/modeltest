"""Data invariant tests: structural guarantees on the validation data."""

from __future__ import annotations

from typing import Any, List, Optional

from modeltest.core.base import ModelTest, TestContext


class DataInvariantTest(ModelTest):
    """Assert expected columns are present (and types match, if given)."""

    def __init__(
        self,
        expected_columns: Optional[List[str]] = None,
        max_null_ratio: float = 0.02,
    ):
        self.expected_columns = expected_columns or []
        self.max_null_ratio = max_null_ratio

    def test(self, ctx: TestContext) -> Any:
        X = ctx.X_val
        if self.expected_columns:
            missing = [c for c in self.expected_columns if c not in X.columns]
            assert not missing, f"Missing columns: {missing}"

        if self.max_null_ratio >= 0:
            null_ratio = X.isna().mean()
            heavy = null_ratio[null_ratio > self.max_null_ratio]
            assert heavy.empty, (
                f"Columns exceed null ratio {self.max_null_ratio}: {dict(heavy)}"
            )


class NoNullTest(ModelTest):
    """Assert no null values exist in the validation data."""

    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns

    def test(self, ctx: TestContext) -> Any:
        X = ctx.X_val if self.columns is None else ctx.X_val[self.columns]
        assert not X.isna().any().any(), "Null values found in validation data"
