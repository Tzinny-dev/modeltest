"""Tests for drift scenarios (PSI / KS)."""
import numpy as np
import pandas as pd
import pytest

from modeltest import ModelTest, TestContext
from modeltest.core.base import TestStatus
from modeltest.scenarios import DataDriftTest, KSTest
from modeltest.scenarios._utils import ks_statistic, psi


def _ctx(
    train: pd.DataFrame, val: pd.DataFrame, model=None
) -> TestContext:
    return TestContext(model=model, X_train=train, X_val=val, y_val=val.index)


def _df(samples_by_col):
    """Build a DataFrame from {col: array}."""
    return pd.DataFrame(samples_by_col)


class TestPsiFunction:
    def test_identical_distributions_near_zero(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 10000)
        assert psi(data, data) < 0.05

    def test_shifted_distribution_is_large(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 10000)
        b = rng.normal(3, 1, 10000)  # big mean shift
        assert psi(a, b) > 0.5

    def test_handles_constant_column(self):
        val = np.full(100, 7.0)
        assert psi(val, val) == 0.0


class TestKsFunction:
    def test_identical_is_small(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 10000)
        assert ks_statistic(data, data) < 0.05

    def test_shifted_is_large(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 10000)
        b = rng.normal(4, 1, 10000)
        assert ks_statistic(a, b) > 0.5


class TestDataDrift:
    def test_passes_when_no_drift(self):
        rng = np.random.default_rng(0)
        train = _df({"age": rng.normal(45, 12, 5000)})
        val = _df({"age": rng.normal(45, 12, 5000)})
        result = DataDriftTest(feature_cols=["age"], max_psi=0.1).run(_ctx(train, val))
        assert result.status == TestStatus.PASSED

    def test_fails_when_drift(self):
        rng = np.random.default_rng(0)
        train = _df({"age": rng.normal(45, 12, 5000)})
        val = _df({"age": rng.normal(200, 12, 5000)})  # strong mean shift
        result = DataDriftTest(feature_cols=["age"], max_psi=0.1).run(_ctx(train, val))
        assert result.status == TestStatus.FAILED

    def test_uses_all_numeric_cols_by_default(self):
        rng = np.random.default_rng(0)
        train = _df({"age": rng.normal(1, 1, 5000), "income": rng.normal(1, 1, 5000)})
        val = _df({"age": rng.normal(1, 1, 5000), "income": rng.normal(1, 1, 5000)})
        result = DataDriftTest(max_psi=0.1).run(_ctx(train, val))
        assert result.status == TestStatus.PASSED


class TestKS:
    def test_passes_when_no_drift(self):
        rng = np.random.default_rng(0)
        train = _df({"age": rng.normal(45, 12, 5000)})
        val = _df({"age": rng.normal(45, 12, 5000)})
        result = KSTest(feature_cols=["age"], min_p_value=0.05).run(_ctx(train, val))
        assert result.status == TestStatus.PASSED

    def test_fails_when_drift(self):
        rng = np.random.default_rng(0)
        train = _df({"age": rng.normal(45, 12, 5000)})
        val = _df({"age": rng.normal(200, 12, 5000)})
        result = KSTest(feature_cols=["age"], min_p_value=0.05).run(_ctx(train, val))
        assert result.status == TestStatus.FAILED
