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
    cache_predictions: bool = True

    # Per-suite prediction cache shared across every test that runs on this
    # context (run_suite passes the same instance to all tests).
    _cache: Optional[Dict[str, Any]] = field(default=None, repr=False)

    _wrapper: Optional[Any] = field(default=None, repr=False)

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("model_name", type(self.model).__name__))

    def _wrapped(self) -> Any:
        """Return a normalized wrapper around ``self.model`` (lazily built)."""
        if self._wrapper is None:
            from modeltest.wrappers import wrap

            self._wrapper = wrap(self.model)
        return self._wrapper

    def predict(self, X: Any = None) -> Any:
        """Predict with caching.

        Delegates to the model via the framework adapter while transparently
        dropping non-feature columns (via ``model_features``). When
        ``cache_predictions`` is on, the result is keyed by a content hash of
        the input so that multiple tests predicting on the *same data* run the
        model only once per suite.

        Passing ``X=None`` predicts on ``self.X_val``.
        """
        from modeltest.scenarios._utils import model_features

        X = self.X_val if X is None else X
        wrapped = self._wrapped()
        X_feat = model_features(wrapped, X)

        if not self.cache_predictions:
            return wrapped.predict(X_feat)

        if self._cache is None:
            self._cache = {}

        key = self._fingerprint(X_feat)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        pred = wrapped.predict(X_feat)
        self._cache[key] = pred
        return pred

    def predict_proba(self, X: Any = None) -> Any:
        """Probability estimates via the wrapper (or ``None`` if unsupported)."""
        from modeltest.scenarios._utils import model_features

        X = self.X_val if X is None else X
        return self._wrapped().predict_proba(model_features(self._wrapped(), X))

    @staticmethod
    def _fingerprint(X: Any) -> str:
        """Cheap-ish content key for a validation input."""
        import hashlib

        if hasattr(X, "columns") and hasattr(X, "values"):
            # pandas DataFrame / Series: hash column names + row content.
            try:
                from pandas.util import hash_pandas_object

                h = hashlib.sha1()
                h.update("|".join(map(str, X.columns)).encode())
                h.update(hash_pandas_object(X, index=True).values.tobytes())
                return h.hexdigest()
            except Exception:  # noqa: BLE001 - fall back below
                pass
        try:
            import pickle

            return hashlib.sha1(pickle.dumps(X, protocol=4)).hexdigest()
        except Exception:  # noqa: BLE001
            return f"id-{id(X)}"


@dataclass
class TestResult:
    """Outcome of running a single test."""

    __test__ = False  # pytest: an outcome object, not a test hook

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


def assert_metric(
    actual: float, expected: Any, op: Callable[[float, Any], bool], msg: str
) -> None:
    if not op(actual, expected):
        raise AssertionError(msg)
