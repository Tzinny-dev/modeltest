"""Fairness tests: detect unintended bias across protected groups."""

from __future__ import annotations

from typing import Any

import numpy as np

from modeltest.core.base import ModelTest, TestContext
from modeltest.scenarios._utils import (
    group_rates,
    max_pairwise_gap,
    min_max_ratio,
)


class EqualOpportunityTest(ModelTest):
    """Assert that true-positive rates are balanced across protected groups.

    Equal opportunity requires equal TPR across groups for those who actually
    belong to the positive class. The test fails if the gap between the best
    and worst TPR exceeds ``max_diff``.
    """

    def __init__(
        self,
        protected_col: str,
        max_diff: float = 0.1,
        pos_label: int = 1,
    ):
        """Configure the equal-opportunity fairness check.

        Args:
            protected_col: Categorical column in ``X_val`` defining the
                protected groups.
            max_diff: Maximum tolerated gap between the best and worst
                group TPR.
            pos_label: Which class counts as positive.
        """
        self.protected_col = protected_col
        self.max_diff = max_diff
        self.pos_label = pos_label

    def test(self, ctx: TestContext) -> Any:
        """Compute per-group TPRs and assert the gap stays within
        ``max_diff``. The failure detail lists every group's TPR."""
        y_pred = np.asarray(ctx.predict())
        rates = group_rates(ctx.y_val, y_pred, ctx.X_val[self.protected_col])
        tprs = {g: r["tpr"] for g, r in rates.items()}
        gap = max_pairwise_gap(tprs)
        assert gap <= self.max_diff, (
            f"Equal opportunity TPR gap = {gap:.4f} > max_diff {self.max_diff}; "
            f"TPRs: {tprs}"
        )


class StatisticalParityTest(ModelTest):
    """Assert selection (acceptance) rates are balanced across groups.

    Statistical parity / disparate impact compares the fraction of each group
    predicted positive. The test checks *both*:
      - absolute demographic parity difference (``max_diff``)
      - the disparate-impact ratio (``min_ratio``, e.g. 0.8 per the 4/5ths rule)
    """

    def __init__(
        self,
        protected_col: str,
        max_diff: float = 0.1,
        min_ratio: float = 0.8,
        pos_label: int = 1,
    ):
        """Configure the statistical-parity fairness check.

        Args:
            protected_col: Categorical column in ``X_val`` defining the
                protected groups.
            max_diff: Maximum tolerated difference in selection rates.
            min_ratio: Minimum tolerated ratio between the least and most
                selected group (``0.8`` implements the 4/5ths rule).
            pos_label: Which prediction value counts as selected.
        """
        self.protected_col = protected_col
        self.max_diff = max_diff
        self.min_ratio = min_ratio
        self.pos_label = pos_label

    def test(self, ctx: TestContext) -> Any:
        """Compute per-group selection rates and assert both the parity
        difference and the disparate-impact ratio stay within bounds."""
        y_pred = np.asarray(ctx.predict())
        rates = group_rates(ctx.y_val, y_pred, ctx.X_val[self.protected_col])
        sel = {g: r["selection_rate"] for g, r in rates.items()}

        diff = max_pairwise_gap(sel)
        ratio = min_max_ratio(sel)
        assert diff <= self.max_diff, (
            f"Selection-rate difference = {diff:.4f} > max_diff {self.max_diff}; "
            f"rates: {sel}"
        )
        assert ratio >= self.min_ratio, (
            f"Disparate-impact ratio = {ratio:.3f} < min_ratio {self.min_ratio}; "
            f"rates: {sel}"
        )
