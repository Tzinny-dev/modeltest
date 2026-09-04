"""A user-defined custom test, loadable from a YAML suite.

Custom tests subclass ``modeltest.ModelTest`` and override ``test(ctx)``. They
can be referenced from YAML either by registering a name or with a dotted
import path — see ``examples/suite_custom.yaml``.
"""
from __future__ import annotations

from modeltest import ModelTest


class ZeroPredictionShareTest(ModelTest):
    """Sanity guard: the model must predict at least one positive class."""

    def __init__(self, min_positive_share: float = 0.01):
        self.min_positive_share = min_positive_share

    def test(self, ctx) -> None:
        import numpy as np

        shares = np.mean(np.asarray(ctx.predict()) == 1)
        assert shares >= self.min_positive_share, (
            f"positive share {shares:.4f} < min_positive_share "
            f"{self.min_positive_share}"
        )
