# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Documentation site (MkDocs Material) with full usage guides, per-scenario
  references and an API reference generated from docstrings; deployed to
  GitHub Pages on every push to `main` (`mkdocs.yml`, `docs/`,
  `.github/workflows/docs.yml`, `make docs`).
- Complete docstrings for the entire public API (96 objects — core, CLI,
  runners, reports, all 12 scenarios and the model wrappers), rendered in
  the API reference with mkdocstrings.
- Google Analytics 4 for the documentation site, loaded only after the
  visitor grants consent through Material's built-in cookie banner
  (`mkdocs.yml`: `extra.analytics`, `extra.consent`).
- Donation buttons (PayPal and Buy Me a Coffee) in the footer of every
  documentation page, plus GitHub and PyPI icons in the footer social
  links (`docs/overrides/partials/footer.html`, `mkdocs.yml`).
- Versioned documentation with mike (`theme.version: mike`); the Docs
  workflow now runs `mkdocs build --strict` before deploying, so build
  warnings fail CI.

### Changed
- PyPI project links: `Homepage` now points to the documentation site
  (`tzinny-dev.github.io/modeltest`) instead of the GitHub repository,
  which remains under `Repository` (also removed the now-duplicated
  `Documentation` entry).

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
