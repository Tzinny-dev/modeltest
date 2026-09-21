# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-09-20

### Fixed
- JUnit XML reports now emit the standard `<testsuites>` wrapper, as expected
  by CI test reporters (GitHub Actions, Jenkins, GitLab...). The 0.2.0 upload
  accidentally shipped the pre-fix reporter format.
- Package metadata (project URLs) now points to the final repository,
  `github.com/Tzinny-dev/modeltest`, instead of the pre-rename `model-test`.

### Added
- CI matrix: the test suite now runs on Python 3.9–3.13 (previously only 3.12).
- `CHANGELOG.md`, `CONTRIBUTING.md` and `SECURITY.md`.

## [0.2.0] - 2026-09-04

### Added
- First release on PyPI. `modeltest` is a unit-testing framework for ML
  models: define contracts for quality, robustness, fairness and data
  invariants, and run them in CI/CD like pytest for code.
- 12 built-in test scenarios: minimum accuracy, group performance, bootstrap
  confidence thresholds, robustness, data invariants (columns / nulls), PSI
  drift, KS test, equal opportunity, statistical parity (4/5ths rule), SHAP
  feature dominance and top features.
- Multi-framework wrappers: scikit-learn, PyTorch, Keras/TensorFlow, custom
  `ModelWrapper`s and sklearn `Pipeline`s.
- Prediction caching per `suite.run(...)` for fast multi-test suites.
- Declarative YAML suites; custom tests via dotted import path or
  programmatic registration.
- CLI `modeltest validate` with JUnit XML output for CI reporters.
- MLflow integration (`log_suite_result`) via the `modeltest[mlflow]` extra.
- CI-ready GitHub Actions workflows (lint, tests, model validation, PyPI
  release via Trusted Publishing).
- 159 tests with 100% line coverage; ruff lint/format clean.

[Unreleased]: https://github.com/Tzinny-dev/modeltest/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/Tzinny-dev/modeltest/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Tzinny-dev/modeltest/releases/tag/v0.2.0
