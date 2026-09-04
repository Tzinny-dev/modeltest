"""Tests for the CLI (modeltest validate)."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from modeltest.cli import _load_suite, main
from modeltest.core.base import ModelSuite


def _make_artifacts(tmp_path):
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame(
        {"age": rng.normal(45, 12, n), "income": rng.normal(60000, 20000, n)}
    )
    y = (X["income"] > X["age"] * 800 + rng.normal(0, 16000, n)).astype(int)
    df = X.copy()
    df["target"] = y

    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    model_path = tmp_path / "model.pkl"
    data_path = tmp_path / "data.csv"
    joblib.dump(model, model_path)
    df.to_csv(data_path, index=False)
    return str(model_path), str(data_path)


@pytest.fixture
def artifacts(tmp_path):
    return _make_artifacts(tmp_path)


def test_load_suite_python_module(tmp_path):
    suite_file = tmp_path / "suite.py"
    suite_file.write_text(
        "from modeltest import ModelSuite\n"
        "from modeltest.scenarios import MinimumAccuracyTest\n"
        "suite = ModelSuite(name='py')\n"
        "suite.add_test(MinimumAccuracyTest(threshold=0.1))\n"
    )
    suite = _load_suite(str(suite_file))
    assert isinstance(suite, ModelSuite)
    assert suite.name == "py"


def test_load_suite_yaml(tmp_path):
    suite_file = tmp_path / "suite.yaml"
    suite_file.write_text(
        "suite:\n  name: ya\n  tests:\n"
        "    - type: minimum_accuracy\n      params: {threshold: 0.1}\n"
    )
    suite = _load_suite(str(suite_file))
    assert suite.name == "ya"
    assert len(suite.tests) == 1


def test_validate_end_to_end(artifacts, tmp_path):
    model_path, data_path = artifacts
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        "suite:\n  name: cli\n  tests:\n"
        "    - type: minimum_accuracy\n      params: {threshold: 0.1}\n"
    )
    out = tmp_path / "report.xml"
    code = main(
        [
            "validate",
            "--suite",
            str(suite_path),
            "--model",
            model_path,
            "--data",
            data_path,
            "--target",
            "target",
            "--output",
            str(out),
        ]
    )
    assert code == 0
    assert out.exists()
    assert "testsuite" in out.read_text()


def test_validate_failure_exit_code(artifacts, tmp_path):
    model_path, data_path = artifacts
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        "suite:\n  name: cli\n  tests:\n"
        "    - type: minimum_accuracy\n      params: {threshold: 1.5}\n"
    )
    code = main(
        [
            "validate",
            "--suite",
            str(suite_path),
            "--model",
            model_path,
            "--data",
            data_path,
            "--target",
            "target",
        ]
    )
    assert code == 1


def test_validate_train_data_for_drift(artifacts, tmp_path):
    model_path, data_path = artifacts
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        "suite:\n  name: cli\n  tests:\n"
        "    - type: data_drift\n      params: {features: [age, income], max_psi: 5}\n"
    )
    # Use the same file as train + validation (no drift) to pass.
    code = main(
        [
            "validate",
            "--suite",
            str(suite_path),
            "--model",
            model_path,
            "--data",
            data_path,
            "--target",
            "target",
            "--train-data",
            data_path,
        ]
    )
    assert code == 0


def test_main_guard_runs_as_script(artifacts, tmp_path, monkeypatch):
    import runpy
    import sys

    model_path, data_path = artifacts
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        "suite:\n  name: cli\n  tests:\n"
        "    - type: minimum_accuracy\n      params: {threshold: 0.1}\n"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "modeltest",
            "validate",
            "--suite",
            str(suite_path),
            "--model",
            model_path,
            "--data",
            data_path,
            "--target",
            "target",
        ],
    )
    import modeltest.cli as cli_mod

    with pytest.raises(SystemExit) as exc:
        runpy.run_path(cli_mod.__file__, run_name="__main__")
    assert exc.value.code == 0
