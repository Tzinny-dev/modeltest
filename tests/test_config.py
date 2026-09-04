"""Tests for YAML suite loading/serialisation."""

import textwrap

import pytest

from modeltest.config import dump_suite_yaml, load_suite_yaml
from modeltest.core.base import ModelSuite
from modeltest.scenarios import (
    DataDriftTest,
    EqualOpportunityTest,
    MinimumAccuracyTest,
    RobustnessTest,
)


@pytest.fixture
def suite_cfg(tmp_path):
    path = tmp_path / "suite.yaml"
    path.write_text(
        textwrap.dedent(
            """
            suite:
              name: "Test Suite"
              tests:
                - type: minimum_accuracy
                  params: {threshold: 0.85}
                - type: robustness
                  params: {noise_std: 0.01, max_drop: 0.03, metric: accuracy}
                - type: data_drift
                  params: {features: [age, income], max_psi: 0.15}
                - type: equal_opportunity
                  params: {protected: gender, max_diff: 0.1}
            """
        )
    )
    return path


def test_loads_basic_suite(suite_cfg):
    suite = load_suite_yaml(str(suite_cfg))
    assert isinstance(suite, ModelSuite)
    assert suite.name == "Test Suite"
    assert len(suite.tests) == 4
    assert isinstance(suite.tests[0], MinimumAccuracyTest)
    assert isinstance(suite.tests[1], RobustnessTest)
    assert isinstance(suite.tests[2], DataDriftTest)
    assert isinstance(suite.tests[3], EqualOpportunityTest)


def test_aliases_mapped(suite_cfg):
    suite = load_suite_yaml(str(suite_cfg))
    drift = suite.tests[2]
    assert drift.feature_cols == ["age", "income"]
    opp = suite.tests[3]
    assert opp.protected_col == "gender"


def test_unknown_type_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        textwrap.dedent(
            """
            suite:
              name: "bad"
              tests:
                - type: does_not_exist
                  params: {}
            """
        )
    )
    with pytest.raises(ValueError, match="Unknown test type"):
        load_suite_yaml(str(path))


def test_unknown_param_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        textwrap.dedent(
            """
            suite:
              name: "bad"
              tests:
                - type: robustness
                  params: {noise_std: 1, typo_param: 2}
            """
        )
    )
    with pytest.raises(ValueError, match="Unknown params"):
        load_suite_yaml(str(path))


def test_dump_round_trip(tmp_path):
    suite = ModelSuite(name="RT")
    suite.add_tests(
        MinimumAccuracyTest(threshold=0.9),
        RobustnessTest(noise_std=0.02),
    )
    out = tmp_path / "out.yaml"
    dump_suite_yaml(suite, str(out))
    loaded = load_suite_yaml(str(out))
    assert loaded.name == "RT"
    assert len(loaded.tests) == 2
    assert isinstance(loaded.tests[0], MinimumAccuracyTest)
    assert isinstance(loaded.tests[1], RobustnessTest)
    assert loaded.tests[0].threshold == 0.9


class TestConfigErrorPaths:
    def test_missing_suite_mapping(self, tmp_path):
        path = tmp_path / "no_suite.yaml"
        path.write_text("not_suite: true\n")
        with pytest.raises(ValueError, match="top-level"):
            load_suite_yaml(str(path))

    def test_test_entry_not_a_mapping(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text('suite:\n  name: "x"\n  tests:\n    - "not_a_dict"\n')
        with pytest.raises(ValueError, match="must be a mapping"):
            load_suite_yaml(str(path))

    def test_params_not_a_mapping(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(
            'suite:\n  name: "x"\n  tests:\n    - type: minimum_accuracy\n'
            '      params: "bad"\n'
        )
        with pytest.raises(ValueError, match="must be a mapping"):
            load_suite_yaml(str(path))


class TestImportAttr:
    def test_not_a_modeltest_subclass(self):
        from modeltest.config import _import_attr

        with pytest.raises(TypeError, match="not a ModelTest subclass"):
            _import_attr("collections", "OrderedDict")

    def test_missing_attribute(self):
        from modeltest.config import _import_attr

        with pytest.raises(ValueError, match="has no attribute"):
            _import_attr("modeltest.scenarios", "DoesNotExist")
