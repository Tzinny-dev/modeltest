"""Core primitives: ModelTest, TestContext, TestResult, ModelSuite.

The design mirrors how `pytest` structures tests but is oriented to ML:
each test receives a rich context (model + data + metadata) instead of a
bare function signature. A test *passes* by returning normally and *fails*
by raising an assertion / returning a TestResult with passed=False.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class TestStatus(str, Enum):
    __test__ = False  # don't let pytest collect this as a test class

    PASSED = "PASSED"
    FAILED = "FAILED"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


@dataclass
class TestContext:
    """Everything a test may need at runtime.

    A unified context (rather than a loose `model, X, y` signature) lets tests
    grow (drift needs train vs val; explainability needs metadata) without
    breaking the base API.
    """

    __test__ = False  # don't let pytest collect this as a test class

    model: Any
    X_val: Any
    y_val: Any
    X_train: Optional[Any] = None
    y_train: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("model_name", type(self.model).__name__))


@dataclass
class TestResult:
    """Outcome of running a single test."""

    name: str
    status: TestStatus
    detail: str = ""
    duration_ms: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == TestStatus.PASSED


def _now_ms() -> float:
    import time

    return time.perf_counter() * 1000


class ModelTest:
    """Base class for all model tests.

    Subclasses override `test(self, ctx)`. Raising ``AssertionError`` (or any
    ``assert`` failure) marks the test as FAILED; raising anything else marks
    it as ERROR. Returning a :class:`TestResult` lets a test fully control the
    outcome (useful for warning-only / non-blocking checks).
    """

    name: Optional[str] = None

    def test(self, ctx: TestContext) -> Any:
        raise NotImplementedError

    def run(self, ctx: TestContext) -> TestResult:
        import time

        start = time.perf_counter()
        status, detail, metrics = TestStatus.PASSED, "", {}
        try:
            outcome = self.test(ctx)
            if isinstance(outcome, TestResult):
                return outcome
        except AssertionError as exc:
            status = TestStatus.FAILED
            detail = str(exc)
        except Exception as exc:  # noqa: BLE001 - unknown failures are ERRORs
            status = TestStatus.ERROR
            detail = f"{type(exc).__name__}: {exc}"
        duration = (time.perf_counter() - start) * 1000
        return TestResult(
            name=self.name or type(self).__name__,
            status=status,
            detail=detail,
            duration_ms=duration,
            metrics=metrics,
        )


class ModelSuite:
    """A collection of tests run together against a model + data."""

    def __init__(self, name: str = "suite", tests: Optional[List[ModelTest]] = None):
        self.name = name
        self.tests: List[ModelTest] = list(tests) if tests else []

    def add_test(self, test: ModelTest) -> "ModelSuite":
        self.tests.append(test)
        return self

    def add_tests(self, *tests: ModelTest) -> "ModelSuite":
        self.tests.extend(tests)
        return self

    def run(
        self,
        model: Any,
        X_val: Any,
        y_val: Any,
        X_train: Any = None,
        y_train: Any = None,
        **metadata: Any,
    ) -> "SuiteResult":
        ctx = TestContext(
            model=model,
            X_val=X_val,
            y_val=y_val,
            X_train=X_train,
            y_train=y_train,
            metadata=metadata or {},
        )
        from modeltest.core.runner import run_suite

        return run_suite(self, ctx)


@dataclass
class SuiteResult:
    """Aggregate outcome of running every test in a suite."""

    suite_name: str
    results: List[TestResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def num_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def num_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def report(self, style: str = "table") -> str:
        from modeltest.core.report import render_report

        return render_report(self, style=style)


class _ResultProxy:
    """Thin dict-like for test authors to attach metrics."""

    pass


def assert_metric(actual: float, expected: Any, op: Callable[[float, Any], bool], msg: str) -> None:
    if not op(actual, expected):
        raise AssertionError(msg)
