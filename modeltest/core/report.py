"""Report rendering: console table, JSON, and JUnit XML (for CI/CD)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from modeltest.core.base import SuiteResult, TestResult, TestStatus


def _symbol(r: TestResult) -> str:
    return {
        TestStatus.PASSED: "PASS",
        TestStatus.FAILED: "FAIL",
        TestStatus.ERROR: "ERROR",
        TestStatus.SKIPPED: "SKIP",
    }[r.status]


def render_report(result: SuiteResult, style: str = "table") -> str:
    """Render a ``SuiteResult`` in the requested style.

    Args:
        result: The suite outcome to render.
        style: ``"table"`` (console, default), ``"json"`` or ``"junit"``.

    Returns:
        The rendered report as a string.

    Raises:
        ValueError: Via the JSON encoder path if the payload is not
            serializable (should not happen for standard results).
    """
    if style == "json":
        return to_json(result)
    if style == "junit":
        return to_junit_xml(result)
    return _render_table(result)


def _render_table(result: SuiteResult) -> str:
    lines = [f"Suite: {result.suite_name}", ""]
    lines.append(f"{'STATUS':<8} {'TEST':<35} {'TIME (ms)':<12} DETAIL")
    lines.append("-" * 80)
    for r in result.results:
        lines.append(f"{_symbol(r):<8} {r.name:<35} {r.duration_ms:<12.1f} {r.detail}")
    lines.append("-" * 80)
    lines.append(f"{result.num_passed} passed, {result.num_failed} failed")
    return "\n".join(lines)


def to_json(result: SuiteResult) -> str:
    """Serialize a ``SuiteResult`` to a JSON string.

    Includes the suite name, aggregate counts and one object per test with
    name, status, detail, duration and recorded metrics.

    Args:
        result: The suite outcome to serialize.

    Returns:
        Pretty-printed JSON (2-space indent).
    """
    payload = {
        "suite": result.suite_name,
        "passed": result.passed,
        "num_passed": result.num_passed,
        "num_failed": result.num_failed,
        "tests": [
            {
                "name": r.name,
                "status": r.status.value,
                "detail": r.detail,
                "duration_ms": r.duration_ms,
                "metrics": r.metrics,
            }
            for r in result.results
        ],
    }
    return json.dumps(payload, indent=2)


def to_junit_xml(result: SuiteResult) -> str:
    """Serialize a ``SuiteResult`` to JUnit XML.

    The output wraps everything in the standard ``<testsuites>`` element
    expected by CI test reporters (GitHub Actions, Jenkins, GitLab...):
    FAILED tests become ``<failure>`` nodes, ERROR tests ``<error>``.

    Args:
        result: The suite outcome to serialize.

    Returns:
        The XML document as a string.
    """
    failures = sum(1 for r in result.results if r.status == TestStatus.FAILED)
    errors = sum(1 for r in result.results if r.status == TestStatus.ERROR)
    total = len(result.results)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    xml = ['<?xml version="1.0" encoding="UTF-8"?>']
    xml.append("<testsuites>")
    xml.append(
        f'<testsuite name="{_esc(result.suite_name)}" tests="{total}" '
        f'failures="{failures}" errors="{errors}" skipped="0" '
        f'time="{sum(r.duration_ms for r in result.results) / 1000:.3f}" '
        f'timestamp="{timestamp}">'
    )
    for r in result.results:
        xml.append(
            f'  <testcase classname="modeltest" name="{_esc(r.name)}" '
            f'time="{r.duration_ms / 1000:.3f}">'
        )
        if r.status == TestStatus.FAILED:
            xml.append(f'    <failure message="{_esc(r.detail)}" />')
        elif r.status == TestStatus.ERROR:
            xml.append(f'    <error message="{_esc(r.detail)}" />')
        xml.append("  </testcase>")
    xml.append("</testsuite>")
    xml.append("</testsuites>")
    return "\n".join(xml)


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
