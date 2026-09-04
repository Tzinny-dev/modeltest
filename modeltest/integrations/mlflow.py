"""MLflow integration: log a validation run as an experiment run.

``log_suite_result`` writes a finished :class:`SuiteResult` into an MLflow run:

- one **param** per test recording its status,
- one **metric** per numeric test metric, plus the test's duration,
- aggregate metrics (``num_passed`` / ``num_failed`` / ``passed``),
- the rendered JSON report saved as an **artifact** (``modeltest-report.json``).

MLflow is imported lazily so the modeltest core never depends on it. Install
the integration with ``pip install modeltest[mlflow]``.

.. code-block:: python

    import mlflow
    from modeltest.integrations.mlflow import log_suite_result

    result = suite.run(model, X_val, y_val, X_train=X_train, y_train=y_train)

    with mlflow.start_run():            # log into the active run
        log_suite_result(result)

    log_suite_result(result, run_id="abc123")   # or a specific run
"""

from __future__ import annotations

from typing import Optional

_NUMERIC = (int, float)


class MlflowNotInstalledError(RuntimeError):
    """Raised when MLflow is required but not available."""


def _import_mlflow():
    try:
        import mlflow  # noqa: F401
    except ImportError as exc:
        raise MlflowNotInstalledError(
            "MLflow is not installed. Install it with `pip install modeltest[mlflow]`."
        ) from exc
    return mlflow


def _targets(mlflow, run_id):
    """Return (log_param, log_metric, log_text) writer tuples.

    Without a run_id we target the *active* run via the module API; with a
    run_id we use an ``MlflowClient`` scoped to that run.
    """
    if run_id is None:
        return mlflow.log_param, mlflow.log_metric, mlflow.log_text

    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    return (
        lambda k, v: client.log_param(run_id, k, v),
        lambda k, v: client.log_metric(run_id, k, v),
        # log_text with a client is log_text(run_id, text, artifact_file)
        lambda text, path: client.log_text(run_id, text, path),
    )


def log_suite_result(
    result,
    *,
    run_id: Optional[str] = None,
    param_prefix: str = "",
    metric_prefix: str = "",
    log_artifacts: bool = True,
    flush: bool = True,
) -> None:
    """Log a :class:`SuiteResult` into the active (or a named) MLflow run.

    Parameters
    ----------
    result:
        The :class:`SuiteResult` produced by ``suite.run(...)``.
    run_id:
        If given, log into that run (via ``MlflowClient``). Otherwise log into
        the currently active run (call inside ``mlflow.start_run()``).
    param_prefix, metric_prefix:
        Optional prefixes for the logged names.
    log_artifacts:
        Also write the JSON report as an artifact.
    flush:
        Make logging synchronous before returning.
    """
    mlflow = _import_mlflow()
    log_param, log_metric, log_text = _targets(mlflow, run_id)

    for key, value in (
        ("num_passed", result.num_passed),
        ("num_failed", result.num_failed),
        ("passed", int(result.passed)),
    ):
        log_metric(f"{metric_prefix}{key}", float(value))

    for r in result.results:
        name = f"{param_prefix}{r.name}"
        log_param(f"{name}.status", r.status.value)
        log_metric(f"{metric_prefix}{name}.duration_ms", float(r.duration_ms))
        for mkey, mval in r.metrics.items():
            if isinstance(mval, _NUMERIC):
                log_metric(f"{metric_prefix}{name}.{mkey}", float(mval))
            else:
                log_param(f"{name}.{mkey}", str(mval))

    if log_artifacts:
        log_text(result.report(style="json"), "modeltest-report.json")

    if flush:
        mlflow.flush_async_logging()
