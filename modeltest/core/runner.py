"""Test execution entry points."""

from __future__ import annotations

from modeltest.core.base import ModelSuite, ModelTest, SuiteResult, TestContext


def run_test(test: ModelTest, ctx: TestContext):
    return test.run(ctx)


def run_suite(suite: ModelSuite, ctx: TestContext) -> SuiteResult:
    results = [test.run(ctx) for test in suite.tests]
    return SuiteResult(suite_name=suite.name, results=results)
