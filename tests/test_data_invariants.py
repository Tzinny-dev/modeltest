"""Tests for data invariant scenarios (DataInvariantTest, NoNullTest)."""

import numpy as np
import pandas as pd

from modeltest import TestContext
from modeltest.core.base import TestStatus
from modeltest.scenarios import DataInvariantTest, NoNullTest


def _ctx_from_df(df):
    return TestContext(model=None, X_val=df, y_val=np.zeros(len(df)))


class TestDataInvariantTest:
    def test_pass_when_columns_present(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        t = DataInvariantTest(expected_columns=["a", "b"])
        result = t.run(_ctx_from_df(df))
        assert result.status == TestStatus.PASSED

    def test_fail_when_column_missing(self):
        df = pd.DataFrame({"a": [1, 2]})
        t = DataInvariantTest(expected_columns=["a", "missing"])
        result = t.run(_ctx_from_df(df))
        assert result.status == TestStatus.FAILED
        assert "Missing columns" in result.detail

    def test_no_columns_check_only_nulls(self):
        df = pd.DataFrame({"a": [np.nan, np.nan], "b": [1, 2]})
        t = DataInvariantTest(max_null_ratio=1.0)
        result = t.run(_ctx_from_df(df))
        assert result.status == TestStatus.PASSED

    def test_fail_on_heavy_nulls(self):
        df = pd.DataFrame({"a": [np.nan, np.nan], "b": [1, 2]})
        t = DataInvariantTest(max_null_ratio=0.01)
        result = t.run(_ctx_from_df(df))
        assert result.status == TestStatus.FAILED
        assert "exceed null ratio" in result.detail

    def test_skip_null_check_when_negative_threshold(self):
        df = pd.DataFrame({"a": [np.nan, np.nan]})
        t = DataInvariantTest(max_null_ratio=-1)
        result = t.run(_ctx_from_df(df))
        assert result.status == TestStatus.PASSED


class TestNoNullTest:
    def test_pass_when_no_nulls(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        t = NoNullTest()
        result = t.run(_ctx_from_df(df))
        assert result.status == TestStatus.PASSED

    def test_fail_when_nulls_present(self):
        df = pd.DataFrame({"a": [1, np.nan], "b": [3, 4]})
        t = NoNullTest()
        result = t.run(_ctx_from_df(df))
        assert result.status == TestStatus.FAILED
        assert "Null values" in result.detail

    def test_specific_columns(self):
        df = pd.DataFrame({"a": [1, 2], "b": [np.nan, 4]})
        t = NoNullTest(columns=["a"])
        result = t.run(_ctx_from_df(df))
        assert result.status == TestStatus.PASSED

    def test_specific_columns_fail(self):
        df = pd.DataFrame({"a": [1, 2], "b": [np.nan, 4]})
        t = NoNullTest(columns=["b"])
        result = t.run(_ctx_from_df(df))
        assert result.status == TestStatus.FAILED
