"""Tests for the MLflow integration adapter."""

import pytest

mlflow = pytest.importorskip("mlflow")

from modeltest import ModelSuite  # noqa: E402
from modeltest.integrations.mlflow import (  # noqa: E402
    MlflowNotInstalledError,
    log_suite_result,
)
from modeltest.scenarios import (  # noqa: E402
    ConfidenceThresholdTest,
    MinimumAccuracyTest,
)


@pytest.fixture()
def tracking_uri(tmp_path):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    return uri


def _suite_result():
    suite = ModelSuite(name="mlflow-suite")
    suite.add_tests(
        MinimumAccuracyTest(threshold=0.0),
        ConfidenceThresholdTest(
            metric="accuracy", threshold=0.0, n_boot=50, bound="lower"
        ),
    )

    # A tiny stubbed model is enough for these tests (thresholds pass trivially).
    class _Stub:
        def predict(self, X):
            import numpy as np

            return (np.asarray(X)[:, 0] > 0).astype(int)

        def predict_proba(self, X):
            import numpy as np

            return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

    import numpy as np

    from modeltest import TestContext

    X = np.arange(20).reshape(10, 2) - 5
    y = (X[:, 0] > 0).astype(int)
    ctx = TestContext(model=_Stub(), X_val=X, y_val=y)
    from modeltest.core.runner import run_suite

    return run_suite(suite, ctx)


def test_active_run_logging(tracking_uri):
    result = _suite_result()
    with mlflow.start_run():
        log_suite_result(result)
        run_id = mlflow.active_run().info.run_id
    assert run_id

    run = mlflow.get_run(run_id)
    params = run.data.params
    metrics = run.data.metrics
    assert params["MinimumAccuracyTest.status"] == "PASSED"
    assert "MinimumAccuracyTest.duration_ms" in metrics
    assert metrics["num_passed"] == 2.0
    assert metrics["num_failed"] == 0.0
    assert metrics["passed"] == 1.0


def test_run_id_logging(tracking_uri):
    result = _suite_result()
    with mlflow.start_run() as run:
        run_id = run.info.run_id
    # log after the run has finished, via client + run_id
    log_suite_result(result, run_id=run_id, metric_prefix="val.", flush=True)
    run = mlflow.get_run(run_id)
    assert "val.num_passed" in run.data.metrics


def test_artifact_written(tracking_uri):
    result = _suite_result()
    with mlflow.start_run() as run:
        log_suite_result(result)
        run_id = run.info.run_id
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run_id)]
    assert "modeltest-report.json" in artifacts


def test_missing_mlflow_raises(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _no_mlflow(name, *args, **kwargs):
        if name == "mlflow" or name.startswith("mlflow."):
            raise ImportError("nope")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_mlflow)
    from modeltest.integrations import mlflow as mod

    with pytest.raises(MlflowNotInstalledError):
        mod._import_mlflow()
