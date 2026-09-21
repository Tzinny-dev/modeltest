"""Test execution entry points."""

from __future__ import annotations

from modeltest.core.base import ModelSuite, ModelTest, SuiteResult, TestContext


def run_test(test: ModelTest, ctx: TestContext):
    """Run a single test against a context.

    Args:
        test: The test to execute.
        ctx: Context with model, data and the shared prediction cache.

    Returns:
        The test's ``TestResult``.
    """
    return test.run(ctx)


def run_suite(suite: ModelSuite, ctx: TestContext) -> SuiteResult:
    """Run every test in a suite against one shared context.

    Sharing the context is what makes prediction caching work across
    tests: the first prediction is computed and every later test predicting
    on the same data reuses it.

    Args:
        suite: The suite to execute.
        ctx: The (single) context handed to every test.

    Returns:
        Aggregate outcome with one ``TestResult`` per test, in order.
    """
    results = [test.run(ctx) for test in suite.tests]
    return SuiteResult(suite_name=suite.name, results=results)
