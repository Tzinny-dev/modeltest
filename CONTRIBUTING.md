# Contributing to modeltest

Thanks for your interest! This document explains how to set up a development
environment and the conventions used in this repository.

## Development setup

```bash
git clone git@github.com:Tzinny-dev/modeltest.git
cd modeltest
make install          # pip install -e ".[dev]"
```

Python 3.9 or newer is required (the package supports 3.9–3.13). The dev
extra brings pytest, ruff, shap and mlflow.

## Quality gates

Every change must pass the same gates enforced in CI
(`.github/workflows/validate.yml`):

| Gate | Command | Notes |
| ---- | ------- | ----- |
| Lint | `make lint` | `ruff check modeltest/ tests/` |
| Format | `make format` | CI runs `ruff format --check` |
| Tests | `make test` | pytest with a 80% coverage gate (`--cov-fail-under=80`) |
| Distribution | `python -m build && twine check dist/*` | metadata must be valid |
| Model validation | `make validate` | end-to-end CLI + YAML suite on sample artifacts |

Install the pre-commit hooks once — they run ruff and the whitespace/YAML
checks automatically on every commit:

```bash
make precommit
```

## Adding a built-in test scenario

1. Create the test class in `modeltest/scenarios/`, subclassing
   `modeltest.core.base.ModelTest` and implementing `run(ctx)`.
2. Register the YAML `type` name in the `_REGISTRY` map in
   `modeltest/config.py`.
3. Add unit tests under `tests/`. Keep heavy imports (shap, scipy, torch...)
   lazy — inside `run()` — so the core stays light.
4. Document the new test in the README table and add an entry under
   **Unreleased** in `CHANGELOG.md`.

## Custom tests from other projects

Users can plug tests from their own code without touching modeltest: any
`ModelTest` subclass is referenceable from a YAML suite by dotted import path
(`type: mymodule:MyTest`). If you maintain a custom test, that is all it
needs to be compatible.

## Releasing (maintainers)

1. Move the **Unreleased** section of `CHANGELOG.md` into the new version
   with today's date.
2. Bump `version` in `pyproject.toml`.
3. Commit, create the tag `vX.Y.Z`, then `git push origin main --follow-tags`.
4. The *Release to PyPI* workflow builds, checks and publishes via PyPI
   Trusted Publishing (no token needed); a GitHub Release is created with the
   notes.

## Branch protection

`main` is a protected branch:

- Force pushes and deletions are blocked.
- PR merges require **all** status checks to pass. The required check names
  mirror the job names in `.github/workflows/validate.yml` (e.g.
  `Lint + format`, `Run unit tests (py3.9)`).

If you rename or add a job to that workflow, update the required status
checks too — otherwise mergeable PRs would wait forever on a check that no
longer exists. Settings: *Branches → main → Require status checks*.

Direct pushes to `main` are still allowed for maintainers; the required
checks gate PR merges (e.g. Dependabot), not push access.

## Commit style

Short imperative subject lines (`Add drift test for KS p-value`), one logical
change per commit.
