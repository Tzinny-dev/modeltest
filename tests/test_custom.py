"""Tests for loading user-defined (custom) tests into YAML suites."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from modeltest import ModelTest, register, unregister


class MyCustomTest(ModelTest):
    """A benign custom test that always passes (unless a fail flag is set)."""

    def __init__(self, fail: bool = False):
        self.fail = fail

    def test(self, ctx):
        assert not self.fail, "custom failure"


def _yaml(tmp_path, type_name, params=None):
    import yaml

    doc = {
        "suite": {
            "name": "custom",
            "tests": [{"type": type_name, "params": params or {}}],
        }
    }
    p = tmp_path / "suite.yaml"
    p.write_text(yaml.safe_dump(doc))
    return str(p)


class TestRegister:
    def test_register_then_load(self, tmp_path):
        from modeltest.config import load_suite_yaml

        register("my_custom", MyCustomTest)
        try:
            suite = load_suite_yaml(_yaml(tmp_path, "my_custom", {"fail": False}))
            assert isinstance(suite.tests[0], MyCustomTest)
        finally:
            unregister("my_custom")

    def test_register_rejects_non_modeltest(self):
        with pytest.raises(TypeError):
            register("bad", dict)  # type: ignore[arg-type]

    def test_load_unknown_name_raises(self, tmp_path):
        from modeltest.config import load_suite_yaml

        with pytest.raises(ValueError):
            load_suite_yaml(_yaml(tmp_path, "does_not_exist"))

    def test_unregister_then_unknown(self, tmp_path):
        from modeltest.config import load_suite_yaml

        register("my_custom", MyCustomTest)
        unregister("my_custom")
        with pytest.raises(ValueError):
            load_suite_yaml(_yaml(tmp_path, "my_custom"))


class TestImportPath:
    def test_colon_notation(self, tmp_path):
        from modeltest.config import load_suite_yaml

        type_name = "examples.custom_tests:ZeroPredictionShareTest"
        suite = load_suite_yaml(_yaml(tmp_path, type_name))
        assert type(suite.tests[0]).__name__ == "ZeroPredictionShareTest"

    def test_dotted_notation(self, tmp_path):
        from modeltest.config import load_suite_yaml

        type_name = "examples.custom_tests.ZeroPredictionShareTest"
        suite = load_suite_yaml(_yaml(tmp_path, type_name))
        assert type(suite.tests[0]).__name__ == "ZeroPredictionShareTest"

    def test_missing_module_raises(self, tmp_path):
        from modeltest.config import load_suite_yaml

        with pytest.raises(ValueError):
            load_suite_yaml(_yaml(tmp_path, "no.such.module:Test"))

    def test_missing_attr_raises(self, tmp_path):
        from modeltest.config import load_suite_yaml

        with pytest.raises(ValueError):
            load_suite_yaml(_yaml(tmp_path, "examples.custom_tests:Missing"))


class TestCustomSuiteRun:
    def test_custom_test_runs_in_suite(self, tmp_path):
        from modeltest.config import load_suite_yaml

        yaml_path = _yaml(
            tmp_path,
            "examples.custom_tests:ZeroPredictionShareTest",
            {"min_positive_share": 0.0},
        )
        suite = load_suite_yaml(yaml_path)
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.normal(size=200)})
        y = (X["a"] > 0).astype(int)
        model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
        result = suite.run(model, X, y)
        assert result.num_passed == 1
        assert result.passed
