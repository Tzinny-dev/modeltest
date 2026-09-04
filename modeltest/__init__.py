from modeltest.config import register, unregister
from modeltest.core.base import (
    ModelSuite,
    ModelTest,
    SuiteResult,
    TestContext,
    TestResult,
)
from modeltest.core.runner import run_suite, run_test

try:
    from importlib.metadata import version as _metadata_version

    __version__ = _metadata_version("modeltest")
except Exception:  # noqa: BLE001 - not installed via pip
    __version__ = "unknown"

__all__ = [
    "ModelTest",
    "ModelSuite",
    "SuiteResult",
    "TestContext",
    "TestResult",
    "run_test",
    "run_suite",
    "register",
    "unregister",
]
