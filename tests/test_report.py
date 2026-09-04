"""Tests for report rendering (table / JSON / JUnit XML)."""

import json

from modeltest.core.base import ModelSuite, SuiteResult, TestResult, TestStatus
from modeltest.core.report import (
    render_report,
    to_json,
    to_junit_xml,
)


def _suite_result():
    return SuiteResult(
        suite_name="fraud",
        results=[
            TestResult(
                name="MinAcc",
                status=TestStatus.PASSED,
                detail="",
                duration_ms=1.5,
            ),
            TestResult(
                name="Fair",
                status=TestStatus.FAILED,
                detail="TPR gap 0.3 > 0.1",
                duration_ms=2.0,
                metrics={"gap": 0.3},
            ),
        ],
    )


class TestRenderTable:
    def test_render_table_reports_counts(self):
        text = render_report(_suite_result(), style="table")
        assert "Suite: fraud" in text
        assert "1 passed, 1 failed" in text
        assert "PASS" in text
        assert "FAIL" in text


class TestReportJson:
    def test_to_json_shape(self):
        doc = json.loads(to_json(_suite_result()))
        assert doc["suite"] == "fraud"
        assert doc["passed"] is False
        assert doc["num_passed"] == 1
        assert doc["num_failed"] == 1
        assert doc["tests"][0]["name"] == "MinAcc"
        assert doc["tests"][1]["status"] == "FAILED"
        assert doc["tests"][1]["metrics"]["gap"] == 0.3

    def test_render_json_style(self):
        doc = json.loads(render_report(_suite_result(), style="json"))
        assert doc["passed"] is False


class TestReportJunit:
    def test_junit_counts_attributes(self):
        xml = to_junit_xml(_suite_result())
        assert 'name="fraud"' in xml
        assert 'failures="1"' in xml
        assert 'tests="2"' in xml

    def test_junit_failure_tags(self):
        xml = to_junit_xml(_suite_result())
        assert "<failure" in xml
        assert "TPR gap 0.3 &gt; 0.1" in xml  # escaped detail

    def test_junit_error_tag(self):
        res = SuiteResult(
            suite_name="s",
            results=[TestResult(name="T", status=TestStatus.ERROR, detail="boom")],
        )
        xml = to_junit_xml(res)
        assert "<error" in xml
        assert "boom" in xml

    def test_render_junit_style(self):
        xml = render_report(_suite_result(), style="junit")
        assert xml.startswith("<?xml")

    def test_render_default_is_table(self):
        assert "passed" in render_report(_suite_result())


class TestSuiteReportConvenience:
    def test_suite_result_report_json(self):
        suite = ModelSuite(name="x")
        suite.tests = [TestResult(name="a", status=TestStatus.PASSED)]
        res = SuiteResult(suite_name="x", results=suite.tests)
        doc = json.loads(res.report(style="json"))
        assert doc["suite"] == "x"
