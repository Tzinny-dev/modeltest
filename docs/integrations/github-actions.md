# GitHub Actions

`modeltest` is designed to sit in your CI pipeline: after training, validate
the model contract and fail the build if it doesn't hold.

## A complete workflow

The library's own repo runs this pattern on every push and PR (see
[validate.yml](https://github.com/Tzinny-dev/modeltest/blob/main/.github/workflows/validate.yml)):
one job tests the library, another trains a sample model and validates its
contract.

```yaml
name: Model Validation

on: [push, pull_request]

jobs:
  validate-model:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install
        run: pip install modeltest

      - name: Train model (your training pipeline here)
        run: python train.py --output model.pkl

      - name: Validate model contract
        run: |
          modeltest validate \
            --suite suite.yaml \
            --model model.pkl \
            --data validation.csv \
            --target target \
            --train-data train.csv \
            --output reports/model-validation.xml

      - name: Publish test report
        uses: mikepenz/action-junit-report@v5
        if: always()   # report even when the validation step failed
        with:
          report_paths: reports/model-validation.xml
          check_name: Model Contract
```

## How the pieces fit

1. **`modeltest validate` exits `1` when any test fails**, which fails the
   job — your broken contract blocks the merge like any failing test.
2. `--output` writes **JUnit XML** wrapped in a standard `<testsuites>`
   element, so any JUnit reporter renders each model test as its own
   check (name, status, duration, failure message).
3. The report step runs with `if: always()` so you see *which contract*
   failed in the Checks UI, not just a red X.

## Variations

**Nightly drift check** — schedule the same suite against fresh data:

```yaml
on:
  schedule:
    - cron: "0 6 * * *"   # daily, 06:00 UTC
```

**Gate a PR that retrains** — keep the suite file in the repo so contract
changes are reviewed as diffs. Bump a threshold in the same PR that changes
the model, and reviewers see the trade-off explicitly.

**Multiple suites** — one job per suite (e.g. `quality`, `fairness`) with
its own JUnit report gives each contract its own required check.
